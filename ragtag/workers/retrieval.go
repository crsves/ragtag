package workers

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"time"
)

// findPython returns the first Python executable found on PATH, checking the
// RAGTAG_PYTHON env var first, then the platform-appropriate names.
// On Windows "python3" does not exist by default; we fall through to "python"
// and finally "py" (the Windows launcher).
func findPython() string {
	if v := os.Getenv("RAGTAG_PYTHON"); v != "" {
		return v
	}
	for _, name := range []string{"python3", "python", "py"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	return "python3" // fallback: will produce a clear "not found" error on Start()
}

func findProjectPython(ragDir string) string {
	if runtime.GOOS == "windows" {
		candidate := filepath.Join(ragDir, ".venv", "Scripts", "python.exe")
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		return findPython()
	}

	candidate := filepath.Join(ragDir, ".venv", "bin", "python")
	if _, err := os.Stat(candidate); err == nil {
		return candidate
	}

	return findPython()
}

// Chunk represents a retrieved text chunk.
type Chunk struct {
	ChunkID        string `json:"chunk_id"`
	Text           string `json:"text"`
	TimestampStart string `json:"timestamp_start"`
	Sender         string `json:"sender"`
}

// Result is one retrieved document from the retrieval bridge.
type Result struct {
	Rank           int     `json:"rank"`
	Score          float64 `json:"score"`
	RerankScore    float64 `json:"rerank_score"`
	KeywordBoosted bool    `json:"keyword_boosted"`
	IsNeighbor     bool    `json:"is_neighbor"`
	Source         string  `json:"source"`
	Chunk          Chunk   `json:"chunk"`
}

// bridgeRequest is the JSON sent to the bridge subprocess.
type bridgeRequest struct {
	Cmd    string `json:"cmd"`
	Query  string `json:"query,omitempty"`
	K      int    `json:"k,omitempty"`
	Window int    `json:"window,omitempty"`
	Debug  bool   `json:"debug,omitempty"`
	Slug   string `json:"slug,omitempty"`
}

// bridgeResponse is the JSON received from the bridge subprocess.
type bridgeResponse struct {
	Results     []Result               `json:"results"`
	Context     string                 `json:"context"`
	Error       string                 `json:"error"`
	Ready       bool                   `json:"ready"`
	OK          bool                   `json:"ok"`
	Chats       map[string]interface{} `json:"chats"`
	Active      string                 `json:"active"`
	Slug        string                 `json:"slug"`
	DisplayName string                 `json:"display_name"`
}

// Bridge manages the long-running Python subprocess.
type Bridge struct {
	mu      sync.Mutex
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	stdout  *bufio.Scanner
	ready   bool
	readyCh chan struct{} // closed once {"ready": true} is received
	ragDir  string
}

// NewBridge starts the bridge subprocess in ragDir and blocks until the
// {"ready": true} handshake completes or the 30-second timeout fires.
func NewBridge(ragDir string) (*Bridge, error) {
	var cmd *exec.Cmd
	bridgeExecName := "bridge"
	if runtime.GOOS == "windows" {
		bridgeExecName = "bridge.exe"
	}
	bridgePath := filepath.Join(ragDir, bridgeExecName)
	if _, err := os.Stat(bridgePath); err == nil {
		fmt.Fprintf(os.Stderr, "[bridge] using binary: %s\n", bridgePath)
		cmd = exec.Command(bridgePath)
	} else {
		py := findProjectPython(ragDir)
		fmt.Fprintf(os.Stderr, "[bridge] binary not found at %s — falling back to: %s bridge.py\n", bridgePath, py)
		cmd = exec.Command(py, "bridge.py")
	}
	cmd.Dir = ragDir
	cmd.Stderr = os.Stderr
	setProcAttr(cmd)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("bridge stdin pipe: %w", err)
	}

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("bridge stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("bridge start: %w", err)
	}
	fmt.Fprintf(os.Stderr, "[bridge] pid %d started, waiting for ready signal …\n", cmd.Process.Pid)

	b := &Bridge{
		cmd:     cmd,
		stdin:   stdin,
		stdout:  bufio.NewScanner(stdoutPipe),
		readyCh: make(chan struct{}),
		ragDir:  ragDir,
	}

	// Read the ready signal in a goroutine, skipping any non-JSON lines
	// (e.g. library info messages printed to stdout during initialisation).
	go func() {
		for b.stdout.Scan() {
			line := b.stdout.Text()
			if len(line) == 0 || line[0] != '{' {
				fmt.Fprintf(os.Stderr, "[bridge] non-JSON stdout: %s\n", line)
				continue
			}
			var resp bridgeResponse
			if err := json.Unmarshal([]byte(line), &resp); err == nil && resp.Ready {
				b.mu.Lock()
				b.ready = true
				b.mu.Unlock()
				fmt.Fprintf(os.Stderr, "[bridge] ready\n")
				close(b.readyCh)
				return
			}
			fmt.Fprintf(os.Stderr, "[bridge] unexpected line before ready: %s\n", line)
		}
		if err := b.stdout.Err(); err != nil {
			fmt.Fprintf(os.Stderr, "[bridge] stdout read error: %v\n", err)
		} else {
			fmt.Fprintf(os.Stderr, "[bridge] stdout closed before ready signal\n")
		}
	}()

	// Block until ready or timeout.
	select {
	case <-b.readyCh:
		// good
	case <-time.After(30 * time.Second):
		killProc(cmd)
		return nil, fmt.Errorf("bridge did not become ready within 30 s")
	}

	return b, nil
}

// Ready reports whether the bridge has sent its {"ready": true} startup message.
func (b *Bridge) Ready() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.ready
}

// Kill terminates the bridge subprocess.
func (b *Bridge) Kill() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.stdin != nil {
		b.stdin.Close()
	}
	killProc(b.cmd)
}

// sendRaw writes a request and reads the raw JSON response line.
// The mutex is held for the entire round-trip, keeping communication sequential.
func (b *Bridge) sendRaw(req bridgeRequest) ([]byte, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	if _, err := fmt.Fprintf(b.stdin, "%s\n", data); err != nil {
		return nil, fmt.Errorf("write to bridge: %w", err)
	}

	// Skip non-JSON lines (library init messages, log output, etc.)
	for b.stdout.Scan() {
		line := b.stdout.Text()
		if len(line) > 0 && line[0] == '{' {
			return []byte(line), nil
		}
		fmt.Fprintf(os.Stderr, "[bridge] non-JSON stdout: %s\n", line)
	}
	if err := b.stdout.Err(); err != nil {
		return nil, fmt.Errorf("read bridge response: %w", err)
	}
	return nil, fmt.Errorf("bridge stdout closed unexpectedly")
}

// send writes a request and decodes the response into a bridgeResponse.
func (b *Bridge) send(req bridgeRequest) (bridgeResponse, error) {
	raw, err := b.sendRaw(req)
	if err != nil {
		return bridgeResponse{}, err
	}
	var resp bridgeResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return bridgeResponse{}, fmt.Errorf("unmarshal bridge response: %w", err)
	}
	return resp, nil
}

// Retrieve runs a retrieval query and returns the results and assembled context string.
func (b *Bridge) Retrieve(query string, k, window int, debug bool) ([]Result, string, error) {
	resp, err := b.send(bridgeRequest{
		Cmd:    "retrieve",
		Query:  query,
		K:      k,
		Window: window,
		Debug:  debug,
	})
	if err != nil {
		return nil, "", err
	}
	if resp.Error != "" {
		return nil, "", fmt.Errorf("bridge error: %s", resp.Error)
	}
	return resp.Results, resp.Context, nil
}

// ListChats returns all available chats and the active slug.
func (b *Bridge) ListChats() (map[string]interface{}, string, error) {
	resp, err := b.send(bridgeRequest{Cmd: "list_chats"})
	if err != nil {
		return nil, "", err
	}
	if resp.Error != "" {
		return nil, "", fmt.Errorf("bridge error: %s", resp.Error)
	}
	return resp.Chats, resp.Active, nil
}

// SetChat switches the active chat in the bridge.
func (b *Bridge) SetChat(slug string) error {
	resp, err := b.send(bridgeRequest{Cmd: "set_chat", Slug: slug})
	if err != nil {
		return err
	}
	if resp.Error != "" {
		return fmt.Errorf("bridge error: %s", resp.Error)
	}
	if !resp.OK {
		return fmt.Errorf("set_chat failed for slug %q", slug)
	}
	return nil
}

// Stats returns index statistics for the current chat as a generic map so that
// any fields the bridge chooses to send are preserved.
func (b *Bridge) Stats() (map[string]interface{}, error) {
	raw, err := b.sendRaw(bridgeRequest{Cmd: "stats"})
	if err != nil {
		return nil, err
	}

	// Decode into a generic map to capture all fields.
	var m map[string]interface{}
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("unmarshal stats: %w", err)
	}

	if errVal, ok := m["error"].(string); ok && errVal != "" {
		return nil, fmt.Errorf("bridge error: %s", errVal)
	}

	return m, nil
}

// Close shuts down the bridge subprocess gracefully.
func (b *Bridge) Close() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.stdin != nil {
		_ = b.stdin.Close()
	}
	killProc(b.cmd)
}
