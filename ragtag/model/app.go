package model

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	debug2 "runtime/debug"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
	"github.com/emirate/rag-tui/ui"
	"github.com/emirate/rag-tui/workers"
	"github.com/sashabaranov/go-openai"
)

// ─── tea.Msg types ────────────────────────────────────────────────────────────

// BridgeReadyMsg is sent when the Python bridge has initialised.
type BridgeReadyMsg struct{}

// BridgeErrMsg is sent when the bridge fails to start.
type BridgeErrMsg struct{ Err error }

// BridgeRestartedMsg is sent when the bridge has been restarted after a crash.
type BridgeRestartedMsg struct{ Bridge *workers.Bridge }

// RetrievalDoneMsg carries retrieval results back to the event loop.
type RetrievalDoneMsg struct {
	Results    []workers.Result
	Context    string
	DebugStats *workers.DebugStats
	Err        error
}

// StreamTokenMsg carries one streaming token.
type StreamTokenMsg struct{ Token string }

// StreamDoneMsg signals end of streaming (Err may be nil on success).
type StreamDoneMsg struct{ Err error }

// IngestProgressMsg carries a progress update from a running ingest operation.
type IngestProgressMsg struct {
	Pct     int
	Message string
	ch      chan workers.IngestProgress // closed when ingest completes
}

// AgentLLMDoneMsg carries the parsed action from one agentic LLM call.
type AgentLLMDoneMsg struct {
	Action  string // "SEARCH" or "ANSWER"
	Payload string
	Raw     string
}

// AgentRetrievalDoneMsg carries retrieval results for one agentic search step.
type AgentRetrievalDoneMsg struct {
	Query      string
	NumResults int
	TopScore   float64
	ChunkText  string
	Results    []workers.Result // full results for sources display
	DebugStats *workers.DebugStats
	Err        error
}

// ChatListMsg carries the list of available chats.
type ChatListMsg struct {
	Chats  map[string]interface{}
	Active string
	Err    error
}

// StatsMsg carries index statistics.
type StatsMsg struct {
	Stats map[string]interface{}
	Err   error
}

// CheckResultMsg carries the result of a "what to export next" check.
type CheckResultMsg struct {
	Data map[string]interface{}
	Err  error
}

// UpdateMsg carries the result of a self-update attempt.
type UpdateMsg struct {
	Err error
}

// ChatFileListMsg carries the list of raw chat JSON files.
type ChatFileListMsg struct {
	Files []string
	Err   error
}

// ChatFileLoadedMsg carries a fully-formatted chat file ready for the viewer.
type ChatFileLoadedMsg struct {
	Title   string
	Content string
	Err     error
}

// ─── Autocomplete commands ───────────────────────────────────────────────────

// cmdSuggestion is one slash-command entry shown in the autocomplete popup.
type cmdSuggestion struct {
	cmd  string
	desc string
	prio int // lower = more common, displayed first
}

// allCommands is the full command list. Priority order reflects expected usage
// frequency: toggles and retrieval tweaks are typed most often.
var allCommands = []cmdSuggestion{
	{"/debug", "toggle debug output", 1},
	{"/sources", "toggle source table in replies", 2},
	{"/agent", "toggle / run agentic mode", 3},
	{"/tools", "show or toggle agent tools", 3},
	{"/clear", "clear the visible chat log", 3},
	{"/pause", "pause/stop current AI response", 3},
	{"/update", "update ragtag to the latest version", 3},
	{"/mode", "set output mode: plain / structured / rich", 4},
	{"/model", "set model by name/number/nickname", 4},
	{"/k", "set final_k (chunks passed to LLM)", 5},
	{"/window", "set context window size", 6},
	{"/settings", "open settings panel", 7},
	{"/chats", "switch between chats", 8},
	{"/help", "open the interactive help browser", 9},
	{"/stats", "show index statistics", 10},
	{"/pipeline", "open pipeline management", 10},
	{"/ingest", "ingest a file — or view live log if one is running", 10},
	{"/check", "show what date range to export next", 10},
	{"/view", "browse raw chat history", 10},
	{"/confident", "toggle confident mode", 11},
	{"/thinking", "toggle thinking mode", 12},
	{"/rag", "toggle RAG-only mode (skip LLM)", 13},
	{"/minresults", "set minimum results floor (e.g. /minresults 5)", 13},
	{"/threshold", "set score threshold (e.g. /threshold 1.5)", 13},
	{"/back", "go back to chat", 13},
	{"/exit", "exit the TUI", 14},
}

type helpSection struct {
	id      string
	title   string
	summary string
}

var helpSections = []helpSection{
	{"getting-started", "Getting Started", "The fast path for asking questions and switching modes."},
	{"commands", "Commands", "What each slash command does, including /clear."},
	{"features", "Features & Modes", "Agent mode, output modes, chunk browser, sources, and updates."},
	{"tips", "Tips & Tricks", "Small habits that make ragtag feel a lot better to use."},
	{"models", "Models & Retrieval", "How defaults, model picking, retrieval, and window size interact."},
}

// allModels is the built-in model catalogue, numbered 1–N for /model N.
var allModels = []string{
	"meta/llama-3.3-70b-instruct",
	"meta/llama-3.1-70b-instruct",
	"meta/llama-3.1-8b-instruct",
	"meta/llama-3.2-3b-instruct",
	"nvidia/llama-3.1-nemotron-70b-instruct",
	"mistralai/mistral-7b-instruct-v0.3",
	"mistralai/mixtral-8x7b-instruct-v0.1",
	"google/gemma-2-9b-it",
	"microsoft/phi-3-mini-128k-instruct",
	"deepseek-ai/deepseek-v3.2",
	"openai/gpt-oss-120b",
}

// defaultNicknames provides built-in short aliases for common models.
var defaultNicknames = map[string]string{
	"deepseek": "deepseek-ai/deepseek-v3.2",
	"gpt":      "openai/gpt-oss-120b",
	"gpt120":   "openai/gpt-oss-120b",
	"llama":    "meta/llama-3.3-70b-instruct",
	"llama70":  "meta/llama-3.3-70b-instruct",
	"llama8":   "meta/llama-3.1-8b-instruct",
	"llama3b":  "meta/llama-3.2-3b-instruct",
	"nemotron": "nvidia/llama-3.1-nemotron-70b-instruct",
	"mistral":  "mistralai/mistral-7b-instruct-v0.3",
	"mixtral":  "mistralai/mixtral-8x7b-instruct-v0.1",
	"gemma":    "google/gemma-2-9b-it",
	"phi":      "microsoft/phi-3-mini-128k-instruct",
}

// settingsRows defines the ordered fields in the Model settings screen.
var settingsRows = []struct {
	key   string
	label string
}{
	{"model", "NIM Model"},
	{"temperature", "Temperature"},
	{"top_p", "Top-P"},
	{"max_tokens", "Max Tokens"},
	{"thinking", "Thinking Mode"},
	{"nick", "Nickname Model"},
}

// apiSettingsRows defines the API settings screen fields.
var apiSettingsRows = []struct {
	key   string
	label string
}{
	{"api_key", "API Key"},
	{"base_url", "Base URL"},
	{"hf_token", "HF Token"},
}

// retrievalSettingsRows defines the fields in the Retrieval settings screen.
var retrievalSettingsRows = []struct {
	key   string
	label string
}{
	{"final_k", "Final K"},
	{"window", "Window"},
	{"min_results", "Min Results"},
	{"score_threshold", "Score Threshold"},
	{"max_context_chars", "Max Context Chars"},
}

// interfaceSettingsRows defines the toggles in the Interface settings screen.
var interfaceSettingsRows = []struct {
	key   string
	label string
}{
	{"debug", "Debug Output"},
	{"sources", "Show Sources"},
	{"agent", "Agent Mode"},
	{"confident", "Confident Mode"},
	{"rag_only", "RAG-only Mode"},
}

// settingsMenuItems defines the top-level settings category menu entries.
var settingsMenuItems = []struct {
	label  string
	screen Screen
	action string // "stats" = run stats command inline
}{
	{"API", ScreenAPISettings, ""},
	{"Model", ScreenSettings, ""},
	{"Retrieval", ScreenRetrievalSettings, ""},
	{"Interface", ScreenInterfaceSettings, ""},
	{"Index Stats", 0, "stats"},
	{"Pipeline", ScreenPipeline, ""},
	{"Browse Chats", ScreenChatList, ""},
	{"Help", ScreenHelp, ""},
}

// pipelineMenuItems defines entries in the pipeline management screen.
var pipelineMenuItems = []struct {
	label  string
	action string
}{
	{"Ingest new export", "ingest"},
	{"What to export next", "check"},
	{"Rebuild index from scratch", "rebuild"},
	{"Test retrieval (no LLM)", "test"},
	{"Show index stats", "stats"},
}

// ─── AppModel ─────────────────────────────────────────────────────────────────

// AppModel is the root Bubbletea model.
type AppModel struct {
	// Layout
	viewport viewport.Model
	textarea textarea.Model
	spinner  spinner.Model
	width    int
	height   int

	// State
	appState   AppState
	screen     Screen
	prevScreen Screen // tracks last screen for transition detection

	// Data
	messages []ChatMessage
	settings *Settings
	tuiState TUIState
	ragDir   string

	// Workers
	bridge    *workers.Bridge
	activeCtx context.Context    // current cancellable context for AI ops
	cancelFn  context.CancelFunc // cancel the activeCtx

	// Streaming buffers
	streamBuf     bytes.Buffer
	streamSources []workers.Result // sources for the message being streamed

	// Channels
	tokenChan chan string
	errChan   chan error

	// Agentic state
	agentSearches   []string
	agentFailed     []string
	agentCtx        bytes.Buffer // accumulated retrieved context (not the Go context.Context)
	agentStep       int
	agentMaxSteps   int // -1 = use default cap
	agentQuestion   string
	agentConsecFail int

	// Settings overlay
	settingsEditing string // empty = list view; otherwise field key being edited
	settingsInput   string

	// Chat switcher overlay
	chatList     map[string]interface{}
	activeChat   string
	chatSlugList []string // ordered slug list for display

	// Misc
	errMsg        string
	bridgeReady   bool
	mdRenderer    *glamour.TermRenderer
	shouldRestart bool // set to true when user picks "restart" after /update

	// Clarify overlay (shown above input row)
	clarify ClarifyState

	// Update notification
	updateAvailable    string // non-empty when a newer version exists
	updateChangelog    string
	startupUpdateShown bool // true once the startup update clarify has been shown
	startupAPIWarned   bool // true once we've warned that AI is on without an API key

	// Autocomplete
	acSuggestions []cmdSuggestion
	acSelected    int
	acActive      bool
	acConsumedKey bool // true = skip textarea/viewport Update for this msg

	// Help pager
	helpVp      viewport.Model
	helpCursor  int
	helpSection string

	// Settings navigation
	settingsCursor          int
	settingsMenuCursor      int
	retrievalSettingsCursor int
	retrievalEditing        string
	retrievalInput          string

	// API settings
	apiSettingsCursor  int
	apiSettingsEditing string
	apiSettingsInput   string

	// Interface settings
	ifaceSettingsCursor int

	// Pipeline screen
	pipelineCursor      int
	pipelineInputMode   bool   // true = capturing text input (query / file path)
	pipelineInput       string // text typed in pipeline input mode
	pipelinePrompt      string // what we're asking for ("query" or "file")
	pipelineFilePicking bool   // true = showing raw/ file picker for ingest
	pipelineFileCursor  int    // cursor within file picker list
	// Ingest config screen — shown after picking a file, before ingesting.
	pipelineIngestConfig bool   // true = showing ingest config/limit screen
	pipelineConfigFile   string // selected file path
	pipelineConfigCursor int    // cursor in preset list
	pipelineConfigLimit  int    // 0 = all
	pipelineConfigAfter  string // ISO date filter, empty = no filter
	pipelineConfigCustom bool   // true = text input open for custom count or date

	// Ingest progress
	ingestInProgress bool
	ingestFilename   string // base filename being ingested (for status bar)
	ingestProgress   int    // 0-100 percentage, 0 means unknown/indeterminate
	ingestMessage    string // last progress message from bridge
	ingestNewSlug    string // derived slug for the switch-chat prompt
	ingestLog        []string // accumulated progress messages for log viewer
	ingestLogVP      viewport.Model // viewport for ScreenIngestLog

	// Chat file viewer
	chatFileList       []string
	chatFileCursor     int
	chatViewVP         viewport.Model
	chatViewTitle      string
	chatViewContent    string // rendered content stored for search
	chatViewSearchMode bool
	chatViewSearchTerm string
	chatViewMatchLines []int
	chatViewMatchIdx   int

	// Chat switcher cursor
	chatCursor int

	// Model picker
	modelCursor      int
	modelCustomMode  bool
	modelCustomInput string

	// Model nicknames: user-defined aliases (nickname → full model ID)
	modelNicknames map[string]string
	nicknamePath   string

	// Nickname editor
	nickStep     int // 0 = pick model, 1 = type name
	nickModelIdx int
	nickInput    string

	// Startup animation
	animFrame int

	// RAG-only chunk browser
	ragBrowseChunks  []workers.Result
	ragBrowseCursor  int
	ragBrowseVP      viewport.Model
	ragBrowseCtxVP   viewport.Model
	ragBrowseCtxOpen bool
}

// crashLog appends a timestamped message to crash.log in the ragDir (or ~/ragtag, ~/raa).
func crashLog(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	home, _ := os.UserHomeDir()
	for _, dir := range []string{
		filepath.Join(home, "raa"),
		filepath.Join(home, "ragtag"),
	} {
		if fi, err := os.Stat(dir); err == nil && fi.IsDir() {
			f, err := os.OpenFile(filepath.Join(dir, "crash.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
			if err == nil {
				fmt.Fprintf(f, "[%s] %s\n", time.Now().Format(time.RFC3339), msg)
				f.Close()
			}
			return
		}
	}
}

// ─── New ─────────────────────────────────────────────────────────────────────

// New constructs the initial AppModel. Bubbletea will call Init() to fire
// the startup commands.
func New() AppModel {
	ragDir := os.Getenv("RAG_DIR")
	if ragDir == "" {
		if exe, err := os.Executable(); err == nil {
			// Release layout: binary sits directly inside ragDir.
			candidate := filepath.Dir(exe)
			if _, err := os.Stat(filepath.Join(candidate, "nim_config.py")); err == nil {
				ragDir = candidate
			} else {
				// Clone layout: binary is one level deeper (ragDir/ragtag/).
				ragDir = filepath.Dir(candidate)
			}
		} else {
			ragDir = "."
		}
	}

	nimConfigPath := filepath.Join(ragDir, "nim_config.py")
	settings, err := LoadSettings(nimConfigPath)
	if err != nil {
		settings, _ = LoadSettings("/dev/null")
		settings.path = nimConfigPath
	}

	tuiStatePath := filepath.Join(ragDir, ".tui_state.json")
	tuiState := LoadTUIState(tuiStatePath)

	nicknamePath := filepath.Join(ragDir, ".model_nicknames.json")
	userNicknames := loadNicknameFile(nicknamePath)

	// Textarea
	ta := textarea.New()
	ta.Placeholder = "Ask a question or type /help…"
	ta.CharLimit = 2000
	ta.SetHeight(1)
	ta.SetWidth(80)
	ta.ShowLineNumbers = false
	ta.Focus()
	ta.KeyMap.InsertNewline.SetEnabled(false)

	// Viewport
	vp := viewport.New(80, 20)
	vp.SetContent("")

	// Help pager viewport (sized at first WindowSizeMsg)
	helpVp := viewport.New(80, 20)

	// Chat file viewer viewport (sized at first WindowSizeMsg)
	chatViewVP := viewport.New(80, 20)

	// Ingest log viewport (sized at first WindowSizeMsg)
	ingestLogVP := viewport.New(80, 20)

	// Spinner
	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = lipgloss.NewStyle().Foreground(ui.ColorYellow)

	// Glamour markdown renderer
	mdRenderer, _ := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithWordWrap(100),
	)

	// Start bridge
	bridge, bridgeErr := workers.NewBridge(ragDir)

	m := AppModel{
		viewport:       vp,
		helpVp:         helpVp,
		chatViewVP:     chatViewVP,
		ingestLogVP:    ingestLogVP,
		textarea:       ta,
		spinner:        sp,
		appState:       StateStarting,
		screen:         ScreenChat,
		settings:       settings,
		tuiState:       tuiState,
		ragDir:         ragDir,
		bridge:         bridge,
		mdRenderer:     mdRenderer,
		agentMaxSteps:  -1,
		tokenChan:      make(chan string, 100),
		errChan:        make(chan error, 1),
		modelNicknames: userNicknames,
		nicknamePath:   nicknamePath,
	}

	if bridgeErr != nil {
		m.appState = StateError
		m.errMsg = fmt.Sprintf("Failed to start retriever: %v", bridgeErr)
		m.messages = append(m.messages, ChatMessage{
			Role:    "error",
			Content: "Failed to start retriever: " + bridgeErr.Error(),
		})
	}

	return m
}

// waitForBridgeCmd polls until the bridge is ready and then returns BridgeReadyMsg.
func waitForBridgeCmd(b *workers.Bridge) tea.Cmd {
	return func() tea.Msg {
		for i := 0; i < 600; i++ { // up to 60 seconds
			if b.Ready() {
				return BridgeReadyMsg{}
			}
			time.Sleep(100 * time.Millisecond)
		}
		return BridgeErrMsg{Err: fmt.Errorf("bridge startup timed out")}
	}
}

// ─── Init ────────────────────────────────────────────────────────────────────

func (m AppModel) Init() tea.Cmd {
	// Fire version check in the background at startup.
	versionCheck := checkVersionCmd()
	if m.bridge == nil {
		return tea.Batch(m.spinner.Tick, animTickCmd(), versionCheck)
	}
	return tea.Batch(m.spinner.Tick, animTickCmd(), waitForBridgeCmd(m.bridge), versionCheck)
}

// ─── Update ──────────────────────────────────────────────────────────────────

func (m AppModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {

	// ── Window resize ──────────────────────────────────────────────────────
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.viewport.Width = msg.Width
		m.viewport.Height = m.viewportHeight()
		m.textarea.SetWidth(msg.Width - 8)
		m.helpVp.Width = m.helpVpWidth()
		m.helpVp.Height = m.helpVpHeight()
		m.refreshHelpContent()
		m.chatViewVP.Width = m.chatViewVpWidth()
		m.chatViewVP.Height = m.chatViewVpHeight()
		m.ingestLogVP.Width = m.helpVpWidth()
		m.ingestLogVP.Height = m.ingestLogVpHeight()
		if m.screen == ScreenRagBrowser {
			m.resizeRagBrowserViewports()
		}
		m.renderMessages()

	// ── Spinner tick ───────────────────────────────────────────────────────
	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		if m.appState != StateIdle || m.ingestInProgress {
			cmds = append(cmds, cmd)
		}

	// ── Animation tick ────────────────────────────────────────────────────
	case AnimTickMsg:
		m.animFrame = (m.animFrame + 1) % len(rabbitFrames)
		if m.appState == StateStarting || len(m.messages) == 0 {
			cmds = append(cmds, animTickCmd())
		}

	// ── Bridge lifecycle ───────────────────────────────────────────────────
	case BridgeReadyMsg:
		m.bridgeReady = true
		m.appState = StateIdle
		m.maybeWarnMissingAPIKey()
		cmds = append(cmds, fetchChatListCmd(m.bridge))

	case BridgeRestartedMsg:
		if m.bridge != nil {
			m.bridge.Kill()
		}
		m.bridge = msg.Bridge
		m.bridgeReady = true
		m.appState = StateIdle
		m.addMessage(ChatMessage{Role: "system", Content: "✓ Bridge restarted — try again."})
		cmds = append(cmds, fetchChatListCmd(m.bridge))

	case BridgeErrMsg:
		m.appState = StateError
		m.errMsg = msg.Err.Error()
		m.addMessage(ChatMessage{Role: "error", Content: "Bridge error: " + msg.Err.Error()})

	// ── Retrieval done ─────────────────────────────────────────────────────
	case RetrievalDoneMsg:
		if msg.Err != nil {
			m.appState = StateIdle
			m.addMessage(ChatMessage{Role: "error", Content: "Retrieval error: " + msg.Err.Error()})
			break
		}
		m.streamSources = msg.Results
		if m.tuiState.Debug {
			keyStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
			valStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite)
			dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
			rankStyle := lipgloss.NewStyle().Foreground(ui.ColorGreen).Bold(true)
			starStyle := lipgloss.NewStyle().Foreground(ui.ColorYellow).Bold(true)

			var sb strings.Builder
			if msg.DebugStats != nil {
				ds := msg.DebugStats
				sb.WriteString(keyStyle.Render("  query_type") + dimStyle.Render(" = ") + valStyle.Render(ds.QueryType) + "\n")
				sb.WriteString(keyStyle.Render("  faiss_k   ") + dimStyle.Render(" = ") + valStyle.Render(fmt.Sprintf("%d", ds.FaissK)) +
					"   " + keyStyle.Render("bm25_k") + dimStyle.Render(" = ") + valStyle.Render(fmt.Sprintf("%d", ds.BM25K)) + "\n")
				sb.WriteString(keyStyle.Render("  FAISS hits") + dimStyle.Render(" = ") + valStyle.Render(fmt.Sprintf("%d", ds.FaissHits)) +
					"   " + keyStyle.Render("BM25 hits") + dimStyle.Render(" = ") + valStyle.Render(fmt.Sprintf("%d", ds.BM25Hits)) + "\n")
				sb.WriteString(keyStyle.Render("  merged    ") + dimStyle.Render(" = ") + valStyle.Render(fmt.Sprintf("%d", ds.MergedPool)) +
					"   " + keyStyle.Render("bm25_unique") + dimStyle.Render(" = ") + valStyle.Render(fmt.Sprintf("%d", ds.BM25Unique)) + "\n")
				sb.WriteString(keyStyle.Render("  neighbors ") + dimStyle.Render(" = ") + valStyle.Render(fmt.Sprintf("%d", ds.NeighborsAdded)) +
					"   " + keyStyle.Render("candidates") + dimStyle.Render(" = ") + valStyle.Render(fmt.Sprintf("%d", ds.TotalCandidates)) + "\n")
				sb.WriteString(keyStyle.Render("  reranked  ") + dimStyle.Render(" → ") + valStyle.Render(fmt.Sprintf("top %d", ds.Reranked)) + "\n")
			} else {
				sb.WriteString(keyStyle.Render("  retrieved") + dimStyle.Render(" = ") +
					valStyle.Render(fmt.Sprintf("%d chunks, %d chars context", len(msg.Results), len(msg.Context))) + "\n")
			}
			m.addMessage(ChatMessage{Role: "system", Content: "debug · retrieval stats\n" + sb.String(), Prerendered: true})

			// Results table — only shown in debug mode when ShowSources is off;
			// if sources are enabled, the per-answer source table covers this.
			if !m.tuiState.ShowSources {
				limit := len(msg.Results)
				if limit > 10 {
					limit = 10
				}
				if limit > 0 {
					var rb strings.Builder
					rb.WriteString(dimStyle.Render(fmt.Sprintf("  %-3s  %-7s  %-3s  %-8s  %-19s  %s", "#", "score", "★", "src", "timestamp", "text")) + "\n")
					rb.WriteString(dimStyle.Render("  "+strings.Repeat("─", 80)) + "\n")
					for i, r := range msg.Results[:limit] {
						score := r.RerankScore
						if score == 0 {
							score = r.Score
						}
						star := " "
						if r.KeywordBoosted {
							star = starStyle.Render("★")
						}
						src := r.Source
						if len(src) > 7 {
							src = src[:7]
						}
						ts := r.Chunk.TimestampStart
						if len(ts) > 19 {
							ts = ts[:19]
						}
						text := r.Chunk.Text
						maxText := 55
						if len(text) > maxText {
							text = text[:maxText] + "…"
						}
						rb.WriteString(
							rankStyle.Render(fmt.Sprintf("  %2d", i+1)) +
								valStyle.Render(fmt.Sprintf("  %7.4f  ", score)) +
								star +
								valStyle.Render(fmt.Sprintf("  %-8s %-19s  %s", src, ts, text)) + "\n")
					}
					m.addMessage(ChatMessage{Role: "system", Content: "debug · top results\n" + rb.String(), Prerendered: true})
				}
			}
		}
		cmds = append(cmds, m.startStreamingCmd(msg.Context)...)

	// ── Streaming tokens ───────────────────────────────────────────────────
	case StreamTokenMsg:
		m.streamBuf.WriteString(msg.Token)
		m.renderStreamingMessage()
		cmds = append(cmds, waitForTokenCmd(m.tokenChan, m.errChan))

	case StreamDoneMsg:
		content := m.streamBuf.String()
		m.streamBuf.Reset()
		m.appState = StateIdle
		// Replace the in-progress streaming message with the final one.
		if len(m.messages) > 0 && m.messages[len(m.messages)-1].Role == "assistant" {
			m.messages = m.messages[:len(m.messages)-1]
		}
		if content != "" {
			// Try clarify mode first.
			if cs, ok := ParseClarifyOutput(content); ok {
				cs.OriginalQuery = m.lastUserQuery()
				m.clarify = *cs
				m.renderMessages()
				return m, tea.Batch(cmds...)
			}
			// Try to parse and render rich/structured JSON output.
			prerendered := false
			if rendered, ok := ParseRichOutput(content, m.width); ok {
				content = rendered
				prerendered = true
			}
			finalMsg := ChatMessage{
				Role:        "assistant",
				Content:     content,
				Sources:     m.streamSources,
				Prerendered: prerendered,
			}
			m.addMessage(finalMsg)
			// Populate chunk browser so B works after any LLM answer.
			if len(m.streamSources) > 0 {
				m.ragBrowseChunks = make([]workers.Result, len(m.streamSources))
				copy(m.ragBrowseChunks, m.streamSources)
				m.ragBrowseCursor = 0
				m.ragBrowseCtxOpen = false
			}
		}
		m.streamSources = nil
		if msg.Err != nil {
			m.addMessage(ChatMessage{Role: "error", Content: "Stream error: " + msg.Err.Error()})
		}

	// ── Agentic LLM response ───────────────────────────────────────────────
	case AgentLLMDoneMsg:
		workers.Logf(m.ragDir, "agent llm done action=%s payload=%q rawChars=%d", msg.Action, msg.Payload, len(msg.Raw))
		// In debug mode, only show LLM diagnostic when the action adds context
		// not visible from the agentic_step messages (SEARCH queries are already
		// shown; ANSWER content appears in the final assistant message).
		if m.tuiState.Debug && strings.TrimSpace(msg.Raw) != "" {
			action := strings.ToUpper(strings.TrimSpace(msg.Action))
			if action != "SEARCH" && action != "ANSWER" {
				// Unrecognised LLM output — show raw for debugging.
				barStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
				var rawLines []string
				for _, line := range strings.Split(strings.TrimSpace(msg.Raw), "\n") {
					rawLines = append(rawLines, barStyle.Render("  │ ")+line)
				}
				m.addMessage(ChatMessage{
					Role:        "system",
					Content:     "debug · agent llm (unrecognised)\n" + strings.Join(rawLines, "\n"),
					Prerendered: true,
				})
			}
		}
		switch strings.ToUpper(msg.Action) {
		case "SEARCH":
			agentQuery := msg.Payload
			// Deduplicate
			for _, prev := range m.agentSearches {
				if strings.EqualFold(prev, agentQuery) {
					m.addMessage(ChatMessage{Role: "agentic_step", Content: fmt.Sprintf("[dup skip] %s", agentQuery)})
					cmds = append(cmds, m.runAgentStepCmd())
					return m, tea.Batch(cmds...)
				}
			}
			m.agentSearches = append(m.agentSearches, agentQuery)
			m.addMessage(ChatMessage{Role: "agentic_step", Content: fmt.Sprintf("[SEARCH] %s", agentQuery)})
			cmds = append(cmds, agentRetrievalCmd(m.bridge, agentQuery, m.settings.FinalK, m.agentToolWindow(), m.tuiState.Debug, m.tuiState.MinResults, m.tuiState.ScoreThreshold))

		case "ANSWER":
			m.appState = StateIdle
			searchInfo := ""
			if len(m.agentSearches) > 0 {
				searchInfo = fmt.Sprintf("\n\n*(agentic · %d searches)*", len(m.agentSearches))
			}
			m.addMessage(ChatMessage{
				Role:        "assistant",
				Content:     msg.Payload + searchInfo,
				Searches:    m.agentSearches,
				NumSearches: len(m.agentSearches),
				Sources:     m.streamSources, // accumulated from all agent retrieval steps
			})
			if strings.HasPrefix(msg.Payload, "[LLM error:") {
				m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("details logged to %s", filepath.Join(m.ragDir, "ragtag.log"))})
			}
			// Populate chunk browser with all sources from this agent run so B works.
			if len(m.streamSources) > 0 {
				m.ragBrowseChunks = make([]workers.Result, len(m.streamSources))
				copy(m.ragBrowseChunks, m.streamSources)
				m.ragBrowseCursor = 0
				m.ragBrowseCtxOpen = false
			}
			m.resetAgentState()

		default:
			// LLM didn't follow format — treat entire response as an answer.
			m.appState = StateIdle
			m.addMessage(ChatMessage{Role: "assistant", Content: msg.Payload, Sources: m.streamSources})
			m.resetAgentState()
		}

	// ── Agentic retrieval result ───────────────────────────────────────────
	case AgentRetrievalDoneMsg:
		if msg.Err != nil {
			workers.Logf(m.ragDir, "agent retrieval error query=%q err=%v", msg.Query, msg.Err)
			m.addMessage(ChatMessage{Role: "error", Content: "Agent retrieval error: " + msg.Err.Error()})
			m.appState = StateIdle
			m.resetAgentState()
			break
		}
		workers.Logf(m.ragDir, "agent retrieval query=%q results=%d top=%.3f chunkChars=%d", msg.Query, msg.NumResults, msg.TopScore, len(msg.ChunkText))

		// Accumulate retrieval results for /sources display on the final answer.
		m.streamSources = append(m.streamSources, msg.Results...)

		usefulResults := msg.NumResults > 0 && strings.TrimSpace(msg.ChunkText) != ""
		resultState := "✓"
		if !usefulResults {
			resultState = "✗"
		}

		if m.tuiState.Debug {
			keyStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
			valStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite)
			dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
			var sb strings.Builder
			// Consolidated one-line summary + optional stats block.
			sb.WriteString(keyStyle.Render("  results") + dimStyle.Render(" = ") +
				valStyle.Render(fmt.Sprintf("%d  top=%.3f  %s", msg.NumResults, msg.TopScore, resultState)) + "\n")
			if msg.DebugStats != nil {
				ds := msg.DebugStats
				sb.WriteString(keyStyle.Render("  FAISS") + dimStyle.Render(" = ") + valStyle.Render(fmt.Sprintf("%d", ds.FaissHits)) +
					"   " + keyStyle.Render("BM25") + dimStyle.Render(" = ") + valStyle.Render(fmt.Sprintf("%d", ds.BM25Hits)) +
					"   " + keyStyle.Render("reranked") + dimStyle.Render(" → ") + valStyle.Render(fmt.Sprintf("top %d", ds.Reranked)) + "\n")
			}
			m.addMessage(ChatMessage{Role: "system", Content: "debug · agent retrieval\n" + sb.String(), Prerendered: true})
		} else {
			m.addMessage(ChatMessage{
				Role:    "agentic_step",
				Content: fmt.Sprintf("[retrieved %d chunks, top=%.3f]", msg.NumResults, msg.TopScore),
			})
		}

		if !usefulResults {
			m.agentFailed = append(m.agentFailed, msg.Query)
			m.agentConsecFail++
			m.agentCtx.WriteString(fmt.Sprintf("\n\n--- '%s' — low confidence, try different angle ---", msg.Query))
			workers.Logf(m.ragDir, "agent context rejected query=%q consecFail=%d", msg.Query, m.agentConsecFail)
		} else {
			m.agentConsecFail = 0
			m.agentCtx.WriteString(fmt.Sprintf("\n\n--- Results for: '%s' ---\n%s", msg.Query, msg.ChunkText))
			workers.Logf(m.ragDir, "agent context accepted query=%q contextChars=%d", msg.Query, len(msg.ChunkText))
		}

		m.agentStep++

		if m.agentConsecFail >= 3 || m.agentStep >= m.agentHardCap() {
			workers.Logf(m.ragDir, "agent forcing final answer step=%d consecFail=%d hardCap=%d", m.agentStep, m.agentConsecFail, m.agentHardCap())
			m.addMessage(ChatMessage{Role: "agentic_step", Content: "[forcing final answer…]"})
			cmds = append(cmds, m.forceAgentAnswerCmd())
		} else {
			workers.Logf(m.ragDir, "agent continuing to next step=%d", m.agentStep+1)
			cmds = append(cmds, m.runAgentStepCmd())
		}

	// ── Chat list ──────────────────────────────────────────────────────────
	case ChatListMsg:
		if msg.Err == nil {
			m.chatList = msg.Chats
			m.activeChat = msg.Active
			m.chatSlugList = sortedSlugs(msg.Chats)
		}

	// ── Stats ──────────────────────────────────────────────────────────────
	case StatsMsg:
		if msg.Err != nil {
			m.addMessage(ChatMessage{Role: "error", Content: "Stats error: " + msg.Err.Error()})
		} else {
			var sb strings.Builder
			sb.WriteString("Index statistics:\n")
			for k, v := range msg.Stats {
				sb.WriteString(fmt.Sprintf("  %s: %v\n", k, v))
			}
			m.addMessage(ChatMessage{Role: "system", Content: sb.String()})
		}

	// ── Ingest progress ────────────────────────────────────────────────────
	case IngestProgressMsg:
		if msg.Pct >= 0 {
			m.ingestProgress = msg.Pct
			m.ingestMessage = msg.Message
		}
		if msg.Message != "" {
			var line string
			if msg.Pct < 0 {
				// Log-only line from print() capture — indent as detail
				line = fmt.Sprintf("       ↳ %s", msg.Message)
			} else {
				line = fmt.Sprintf("[%3d%%] %s", msg.Pct, msg.Message)
			}
			m.ingestLog = append(m.ingestLog, line)
			m.refreshIngestLog()
		}
		// Re-schedule to read the next progress update from the same channel.
		return m, tea.Batch(m.spinner.Tick, waitForIngestProgressCmd(msg.ch))

	// ── Pipeline results ───────────────────────────────────────────────────
	case PipelineResultMsg:
		m.screen = ScreenChat
		if msg.Err != nil {
			m.ingestInProgress = false
			m.ingestProgress = 0
			m.ingestLog = append(m.ingestLog, fmt.Sprintf("[ERR] %v", msg.Err))
			m.refreshIngestLog()
			m.viewport.Height = m.viewportHeight() // ingest bar gone, restore height
			m.ingestLogVP.Height = m.ingestLogVpHeight()
			if isBridgePipeError(msg.Err) {
				// Bridge process died — kill it and restart automatically.
				if m.bridge != nil {
					m.bridge.Kill()
					m.bridge = nil
					m.bridgeReady = false
				}
				m.addMessage(ChatMessage{Role: "error", Content: fmt.Sprintf("⚠ Bridge crashed during %s — restarting...", msg.Action)})
				cmds = append(cmds, restartBridgeCmd(m.ragDir))
			} else {
				m.addMessage(ChatMessage{Role: "error", Content: fmt.Sprintf("[pipeline/%s] %v", msg.Action, msg.Err)})
			}
		} else if msg.Action == "ingest" {
			m.ingestInProgress = false
			m.ingestProgress = 100
			m.ingestLog = append(m.ingestLog, "[100%] Done.")
			m.refreshIngestLog()
			m.viewport.Height = m.viewportHeight() // ingest bar gone, restore height
			m.ingestLogVP.Height = m.ingestLogVpHeight()
			// Refresh chat list and offer to switch to the newly indexed chat.
			slug := m.ingestNewSlug
			m.ingestFilename = ""
			m.clarify = ClarifyState{
				Active:   true,
				Kind:     ClarifyKindIngestSwitch,
				Question: fmt.Sprintf("✓ Ingest complete — switch to %q now?", slug),
				Options: []ClarifyOption{
					{ID: "yes", Label: "Switch now"},
					{ID: "no", Label: "Stay here"},
				},
			}
			cmds = append(cmds, fetchChatListCmd(m.bridge))
		} else {
			m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("✓ [%s] %s", msg.Action, msg.Message)})
		}

	// ── Chat file list ─────────────────────────────────────────────────────
	case ChatFileListMsg:
		if msg.Err != nil {
			m.pipelineFilePicking = false
			m.addMessage(ChatMessage{Role: "error", Content: "Could not list raw files: " + msg.Err.Error()})
			m.screen = ScreenChat
		} else {
			m.chatFileList = msg.Files
			if m.pipelineFilePicking {
				m.pipelineFileCursor = 0
			} else {
				m.chatFileCursor = 0
			}
		}

	// ── Check / export-range result ────────────────────────────────────────
	case CheckResultMsg:
		if msg.Err != nil {
			m.addMessage(ChatMessage{Role: "error", Content: "Check error: " + msg.Err.Error()})
		} else {
			accentStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
			dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
			valStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite).Bold(true)
			warnStyle := lipgloss.NewStyle().Foreground(ui.ColorYellow).Bold(true)

			latest, _ := msg.Data["latest"].(string)
			exportAfter, _ := msg.Data["export_after"].(string)
			hoursF, _ := msg.Data["hours_behind"].(float64)
			hours := int(hoursF)

			var sb strings.Builder
			if latest == "" {
				sb.WriteString(warnStyle.Render("⚠  No data in store yet.") + "\n")
				sb.WriteString(dimStyle.Render("   Run a full pipeline rebuild to get started."))
			} else {
				behind := fmt.Sprintf("%dh behind", hours)
				if hours > 48 {
					behind = warnStyle.Render(behind)
				} else {
					behind = dimStyle.Render(behind)
				}
				sb.WriteString(accentStyle.Render("◆ Export range for next ingest") + "\n\n")
				sb.WriteString(dimStyle.Render("  Latest in store : ") + valStyle.Render(latest) + "  " + behind + "\n")
				sb.WriteString(dimStyle.Render("  Export after    : ") + valStyle.Render(exportAfter) + "\n\n")
				sb.WriteString(dimStyle.Render("  DiscordChatExporter flag:\n"))
				sb.WriteString(valStyle.Render(fmt.Sprintf(`    --after "%s"`, exportAfter)) + "\n\n")
				sb.WriteString(dimStyle.Render("  Then drop your export file into ") + valStyle.Render("raw/") + dimStyle.Render(" and run ") + accentStyle.Render("/ingest"))
			}
			m.screen = ScreenChat
			m.addMessage(ChatMessage{Role: "system", Content: sb.String()})
		}

	// ── Self-update ────────────────────────────────────────────────────────
	case UpdateMsg:
		if msg.Err != nil {
			m.addMessage(ChatMessage{Role: "error", Content: "Update failed: " + msg.Err.Error()})
		} else {
			m.clarify = ClarifyState{
				Active:   true,
				Kind:     ClarifyKindRestart,
				Question: "ragtag updated successfully! Restart now to use the new version?",
				Options: []ClarifyOption{
					{ID: "yes", Label: "Restart now"},
					{ID: "no", Label: "Later"},
				},
				Cursor:           0,
				SuggestedDefault: "yes",
			}
		}

	// ── On-demand /update check result ─────────────────────────────────────
	case FreshUpdateCheckMsg:
		if msg.Err != nil {
			m.addMessage(ChatMessage{Role: "error", Content: "Update failed: " + msg.Err.Error()})
		} else if msg.Latest == "" {
			m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("You're already on the latest version (%s).", AppVersion)})
		} else {
			m.updateAvailable = msg.Latest
			m.clarify = ClarifyState{
				Active:   true,
				Kind:     ClarifyKindRestart,
				Question: fmt.Sprintf("ragtag updated to %s! Restart now?", msg.Latest),
				Options: []ClarifyOption{
					{ID: "yes", Label: "Restart now"},
					{ID: "no", Label: "Later"},
				},
				Cursor:           0,
				SuggestedDefault: "yes",
			}
		}

	// ── Silent settings-menu version poll result ────────────────────────────
	case SilentVersionPollMsg:
		if msg.Latest != "" && !m.clarify.Active {
			m.updateAvailable = msg.Latest
			question := fmt.Sprintf("ragtag %s is available — update now?", msg.Latest)
			m.clarify = ClarifyState{
				Active:   true,
				Kind:     ClarifyKindUpdate,
				Question: question,
				Options: []ClarifyOption{
					{ID: "yes", Label: "Update now"},
					{ID: "no", Label: "Later"},
				},
				Cursor:           0,
				SuggestedDefault: "yes",
			}
		}

	// ── Version check ──────────────────────────────────────────────────────
	case VersionCheckMsg:
		if msg.Err == nil && isNewerVersion(AppVersion, msg.Latest) {
			m.updateAvailable = msg.Latest
			m.updateChangelog = msg.Changelog
			// Show startup clarify prompt once, but only when the app is idle
			// (not during agentic mode or streaming — would interrupt the flow).
			if !m.startupUpdateShown && m.appState == StateIdle {
				m.startupUpdateShown = true
				question := fmt.Sprintf("ragtag %s is available (you have %s). Update now?", msg.Latest, AppVersion)
				if msg.Changelog != "" {
					question = fmt.Sprintf("ragtag %s is available — %s. Update now?", msg.Latest, msg.Changelog)
				}
				m.clarify = ClarifyState{
					Active:   true,
					Kind:     ClarifyKindUpdate,
					Question: question,
					Options: []ClarifyOption{
						{ID: "yes", Label: "Update now (/update)"},
						{ID: "no", Label: "Later"},
					},
					Cursor:           0,
					SuggestedDefault: "yes",
				}
			}
		}
		// Schedule next poll in 10 minutes regardless of result.
		cmds = append(cmds, repeatingVersionCheckCmd(10*time.Minute))

	// ── Chat file loaded ───────────────────────────────────────────────────
	case ChatFileLoadedMsg:
		if msg.Err != nil {
			m.addMessage(ChatMessage{Role: "error", Content: "Could not load file: " + msg.Err.Error()})
			m.screen = ScreenChatList
		} else {
			m.chatViewTitle = msg.Title
			m.chatViewContent = msg.Content
			m.chatViewSearchMode = false
			m.chatViewSearchTerm = ""
			m.chatViewMatchLines = nil
			m.chatViewMatchIdx = 0
			m.chatViewVP.Width = m.chatViewVpWidth()
			m.chatViewVP.Height = m.chatViewVpHeight()
			m.chatViewVP.SetContent(msg.Content)
			m.chatViewVP.GotoTop()
			m.screen = ScreenChatViewer
		}

	// ── Key events ─────────────────────────────────────────────────────────
	case tea.KeyMsg:
		// Intercept keys for clarify overlay when active.
		if m.clarify.Active {
			cmds = append(cmds, m.handleClarifyKey(msg)...)
			return m, tea.Batch(cmds...)
		}
		// Global quit
		if msg.Type == tea.KeyCtrlC || msg.String() == "ctrl+q" {
			m.saveState()
			if m.bridge != nil {
				m.bridge.Close()
			}
			return m, tea.Quit
		}

		screenBefore := m.screen
		switch m.screen {
		case ScreenChat:
			cmds = append(cmds, m.handleChatKey(msg)...)
		case ScreenSettings:
			cmds = append(cmds, m.handleSettingsKey(msg)...)
		case ScreenSettingsMenu:
			cmds = append(cmds, m.handleSettingsMenuKey(msg)...)
		case ScreenRetrievalSettings:
			cmds = append(cmds, m.handleRetrievalSettingsKey(msg)...)
		case ScreenAPISettings:
			cmds = append(cmds, m.handleAPISettingsKey(msg)...)
		case ScreenInterfaceSettings:
			cmds = append(cmds, m.handleInterfaceSettingsKey(msg)...)
		case ScreenPipeline:
			cmds = append(cmds, m.handlePipelineKey(msg)...)
		case ScreenChats:
			cmds = append(cmds, m.handleChatsKey(msg)...)
		case ScreenHelp:
			cmds = append(cmds, m.handleHelpKey(msg)...)
		case ScreenModelPicker:
			cmds = append(cmds, m.handleModelPickerKey(msg)...)
		case ScreenNick:
			cmds = append(cmds, m.handleNickKey(msg)...)
		case ScreenChatList:
			cmds = append(cmds, m.handleChatListKey(msg)...)
		case ScreenChatViewer:
			cmds = append(cmds, m.handleChatViewerKey(msg)...)
		case ScreenRagBrowser:
			cmds = append(cmds, m.handleRagBrowserKey(msg)...)
		case ScreenIngestLog:
			cmds = append(cmds, m.handleIngestLogKey(msg)...)
		}
		// Silently poll for updates when user enters the settings menu.
		if m.screen == ScreenSettingsMenu && screenBefore != ScreenSettingsMenu {
			cmds = append(cmds, silentVersionPollCmd())
		}
		m.prevScreen = screenBefore
	}

	// Forward events to textarea only on the main chat screen.
	if m.screen == ScreenChat {
		if !m.acConsumedKey {
			var taCmd tea.Cmd
			m.textarea, taCmd = m.textarea.Update(msg)
			cmds = append(cmds, taCmd)
		}
	}
	// Viewport scroll only in chat.
	if m.screen == ScreenChat && !m.acConsumedKey {
		var vpCmd tea.Cmd
		m.viewport, vpCmd = m.viewport.Update(msg)
		cmds = append(cmds, vpCmd)
	}
	// Chat viewer viewport scroll.
	if m.screen == ScreenChatViewer && !m.acConsumedKey {
		var vpCmd tea.Cmd
		m.chatViewVP, vpCmd = m.chatViewVP.Update(msg)
		cmds = append(cmds, vpCmd)
	}
	m.acConsumedKey = false
	if m.screen == ScreenChat {
		m.updateAutocomplete()
	}

	return m, tea.Batch(cmds...)
}

// ─── Key handlers ─────────────────────────────────────────────────────────────

func (m *AppModel) handleChatKey(msg tea.KeyMsg) []tea.Cmd {
	var cmds []tea.Cmd

	switch msg.Type {
	case tea.KeyEsc:
		// Clear textarea and dismiss autocomplete.
		m.textarea.Reset()
		m.acActive = false
		m.acSuggestions = nil
		m.acSelected = 0
		m.acConsumedKey = true
		return nil

	case tea.KeyTab:
		if m.acActive && len(m.acSuggestions) > 0 {
			// Fill in the currently selected suggestion.
			m.textarea.SetValue(m.acSuggestions[m.acSelected].cmd)
			m.acConsumedKey = true
			return nil
		}

	case tea.KeyUp:
		if m.acActive && len(m.acSuggestions) > 0 {
			m.acSelected = (m.acSelected - 1 + len(m.acSuggestions)) % len(m.acSuggestions)
			m.acConsumedKey = true
			return nil
		}

	case tea.KeyDown:
		if m.acActive && len(m.acSuggestions) > 0 {
			m.acSelected = (m.acSelected + 1) % len(m.acSuggestions)
			m.acConsumedKey = true
			return nil
		}

	case tea.KeyRunes:
		if string(msg.Runes) == "b" && len(m.ragBrowseChunks) > 0 && m.textarea.Value() == "" {
			m.openRagBrowser()
			m.acConsumedKey = true
			return nil
		}

	case tea.KeyEnter:
		query := strings.TrimSpace(m.textarea.Value())
		if query == "" {
			return nil
		}
		m.textarea.Reset()
		m.acActive = false
		m.acSuggestions = nil
		m.acSelected = 0
		m.acConsumedKey = true // prevent textarea from double-processing Enter
		m.screen = ScreenChat  // return from help to chat when submitting

		if strings.HasPrefix(query, "/") {
			cmds = append(cmds, m.handleInlineCmd(query)...)
		} else if m.tuiState.AgentMode {
			cmds = append(cmds, m.startAgentFlow(query, -1)...)
		} else {
			cmds = append(cmds, m.startRetrievalFlow(query)...)
		}

	case tea.KeyPgUp:
		m.viewport.HalfViewUp()
	case tea.KeyPgDown:
		m.viewport.HalfViewDown()
	}

	return cmds
}

// ─── Settings menu ────────────────────────────────────────────────────────────

func (m *AppModel) handleSettingsMenuKey(msg tea.KeyMsg) []tea.Cmd {
	m.acConsumedKey = true
	switch msg.Type {
	case tea.KeyEsc:
		m.screen = ScreenChat
	case tea.KeyUp:
		if m.settingsMenuCursor > 0 {
			m.settingsMenuCursor--
		}
	case tea.KeyDown:
		if m.settingsMenuCursor < len(settingsMenuItems)-1 {
			m.settingsMenuCursor++
		}
	case tea.KeyEnter:
		item := settingsMenuItems[m.settingsMenuCursor]
		if item.action == "stats" {
			m.screen = ScreenChat
			return []tea.Cmd{fetchStatsCmd(m.bridge)}
		}
		if item.screen == ScreenChatList {
			m.screen = ScreenChatList
			m.chatFileCursor = 0
			return []tea.Cmd{loadChatFileListCmd(m.ragDir)}
		}
		m.screen = item.screen
		m.settingsCursor = 0
		m.retrievalSettingsCursor = 0
	}
	return nil
}

func (m AppModel) viewSettingsMenuContent() string {
	var sb strings.Builder
	sb.WriteString(ui.TitleStyle.Render("Settings") + "\n\n")
	for i, item := range settingsMenuItems {
		isCursor := i == m.settingsMenuCursor
		marker := "  "
		label := item.label
		if isCursor {
			marker = ui.ActiveFlagStyle.Render("▶ ")
			label = lipgloss.NewStyle().Foreground(ui.ColorWhite).Render(label)
		} else {
			label = ui.SystemMsgStyle.Render(label)
		}
		sb.WriteString(fmt.Sprintf("%s%s\n", marker, label))
	}
	sb.WriteString("\n")
	sb.WriteString(ui.HelpStyle.Render("↑↓: navigate · Enter: open · ESC: back"))
	return lipgloss.NewStyle().Height(m.viewportHeight()).Render(
		ui.BorderStyle.Width(m.width - 4).Render(sb.String()),
	)
}

// ─── Retrieval settings ───────────────────────────────────────────────────────

func (m *AppModel) handleRetrievalSettingsKey(msg tea.KeyMsg) []tea.Cmd {
	m.acConsumedKey = true
	if m.retrievalEditing != "" {
		switch msg.Type {
		case tea.KeyEsc:
			m.retrievalEditing = ""
			m.retrievalInput = ""
		case tea.KeyEnter:
			m.applyRetrievalEdit()
			m.retrievalEditing = ""
			m.retrievalInput = ""
		case tea.KeyBackspace, tea.KeyDelete:
			if len(m.retrievalInput) > 0 {
				runes := []rune(m.retrievalInput)
				m.retrievalInput = string(runes[:len(runes)-1])
			}
		default:
			if msg.Type == tea.KeyRunes {
				m.retrievalInput += msg.String()
			}
		}
		return nil
	}
	switch msg.Type {
	case tea.KeyEsc:
		m.screen = ScreenSettingsMenu
		m.retrievalSettingsCursor = 0
	case tea.KeyUp:
		if m.retrievalSettingsCursor > 0 {
			m.retrievalSettingsCursor--
		}
	case tea.KeyDown:
		if m.retrievalSettingsCursor < len(retrievalSettingsRows)-1 {
			m.retrievalSettingsCursor++
		}
	case tea.KeyEnter:
		row := retrievalSettingsRows[m.retrievalSettingsCursor]
		m.retrievalEditing = row.key
		switch row.key {
		case "final_k":
			m.retrievalInput = strconv.Itoa(m.settings.FinalK)
		case "window":
			m.retrievalInput = strconv.Itoa(m.tuiState.Window)
		case "min_results":
			m.retrievalInput = strconv.Itoa(m.tuiState.MinResults)
		case "score_threshold":
			m.retrievalInput = fmt.Sprintf("%.3f", m.tuiState.ScoreThreshold)
		case "max_context_chars":
			m.retrievalInput = strconv.Itoa(m.settings.MaxContextChars)
		}
	}
	return nil
}

func (m *AppModel) applyRetrievalEdit() {
	switch m.retrievalEditing {
	case "final_k":
		if v, err := strconv.Atoi(strings.TrimSpace(m.retrievalInput)); err == nil && v > 0 {
			m.settings.FinalK = v
			_ = m.settings.Save()
		}
	case "window":
		if v, err := strconv.Atoi(strings.TrimSpace(m.retrievalInput)); err == nil && v >= 0 {
			m.tuiState.Window = v
			_ = SaveTUIState(filepath.Join(m.ragDir, ".tui_state.json"), m.tuiState)
		}
	case "min_results":
		if v, err := strconv.Atoi(strings.TrimSpace(m.retrievalInput)); err == nil && v >= 0 {
			m.tuiState.MinResults = v
			m.saveState()
		}
	case "score_threshold":
		if v, err := strconv.ParseFloat(strings.TrimSpace(m.retrievalInput), 64); err == nil && v >= 0 {
			m.tuiState.ScoreThreshold = v
			m.saveState()
		}
	case "max_context_chars":
		if v, err := strconv.Atoi(strings.TrimSpace(m.retrievalInput)); err == nil && v > 0 {
			m.settings.MaxContextChars = v
			_ = m.settings.Save()
		}
	}
}

func (m AppModel) viewRetrievalSettingsContent() string {
	var sb strings.Builder
	sb.WriteString(ui.TitleStyle.Render("Retrieval Settings") + "\n\n")
	for i, row := range retrievalSettingsRows {
		isSelected := i == m.retrievalSettingsCursor
		marker := "  "
		numStr := fmt.Sprintf("[%d]", i+1)
		if isSelected {
			marker = ui.ActiveFlagStyle.Render("▶ ")
			numStr = ui.TitleStyle.Render(numStr)
		} else {
			numStr = ui.SystemMsgStyle.Render(numStr)
		}
		var valStr string
		if row.key == m.retrievalEditing {
			valStr = ui.ActiveFlagStyle.Render(m.retrievalInput + "_")
		} else {
			var val string
			switch row.key {
			case "final_k":
				val = strconv.Itoa(m.settings.FinalK)
			case "window":
				val = strconv.Itoa(m.tuiState.Window)
			case "min_results":
				val = strconv.Itoa(m.tuiState.MinResults)
			case "score_threshold":
				val = fmt.Sprintf("%.3f", m.tuiState.ScoreThreshold)
			case "max_context_chars":
				val = strconv.Itoa(m.settings.MaxContextChars)
			}
			if isSelected {
				valStr = lipgloss.NewStyle().Foreground(ui.ColorWhite).Render(val)
			} else {
				valStr = ui.SystemMsgStyle.Render(val)
			}
		}
		sb.WriteString(fmt.Sprintf("%s%s %-20s  %s\n", marker, numStr, row.label, valStr))
	}
	sb.WriteString("\n")
	if m.retrievalEditing != "" {
		sb.WriteString(ui.HelpStyle.Render("Enter to save · ESC to cancel"))
	} else {
		sb.WriteString(ui.HelpStyle.Render("↑↓: navigate · Enter: edit · ESC: back"))
	}
	return lipgloss.NewStyle().Height(m.viewportHeight()).Render(
		ui.BorderStyle.Width(m.width - 4).Render(sb.String()),
	)
}

// ─── API settings ─────────────────────────────────────────────────────────────

func (m *AppModel) handleAPISettingsKey(msg tea.KeyMsg) []tea.Cmd {
	m.acConsumedKey = true
	if m.apiSettingsEditing != "" {
		switch msg.Type {
		case tea.KeyEsc:
			m.apiSettingsEditing = ""
			m.apiSettingsInput = ""
		case tea.KeyEnter:
			m.applyAPISettingsEdit()
			m.apiSettingsEditing = ""
			m.apiSettingsInput = ""
		case tea.KeyBackspace, tea.KeyDelete:
			if len(m.apiSettingsInput) > 0 {
				runes := []rune(m.apiSettingsInput)
				m.apiSettingsInput = string(runes[:len(runes)-1])
			}
		default:
			if msg.Type == tea.KeyRunes {
				m.apiSettingsInput += msg.String()
			}
		}
		return nil
	}
	switch msg.Type {
	case tea.KeyEsc:
		m.screen = ScreenSettingsMenu
	case tea.KeyUp:
		if m.apiSettingsCursor > 0 {
			m.apiSettingsCursor--
		}
	case tea.KeyDown:
		if m.apiSettingsCursor < len(apiSettingsRows)-1 {
			m.apiSettingsCursor++
		}
	case tea.KeyEnter:
		row := apiSettingsRows[m.apiSettingsCursor]
		m.apiSettingsEditing = row.key
		switch row.key {
		case "api_key":
			m.apiSettingsInput = ""
		case "base_url":
			m.apiSettingsInput = m.settings.NIMBaseURL
		case "hf_token":
			m.apiSettingsInput = ""
		}
	}
	return nil
}

func (m *AppModel) applyAPISettingsEdit() {
	switch m.apiSettingsEditing {
	case "api_key":
		if v := strings.TrimSpace(m.apiSettingsInput); v != "" {
			// Strip accidental surrounding brackets (common paste artifact).
			v = strings.Trim(v, "[]")
			m.settings.NIMAPIKey = v
			_ = m.settings.Save()
		}
	case "base_url":
		if v := strings.TrimSpace(m.apiSettingsInput); v != "" {
			m.settings.NIMBaseURL = v
			_ = m.settings.Save()
		}
	case "hf_token":
		if v := strings.TrimSpace(m.apiSettingsInput); v != "" {
			v = strings.Trim(v, "[]")
			m.settings.HFToken = v
			_ = m.settings.Save()
		}
	}
}

func (m AppModel) viewAPISettingsContent() string {
	var sb strings.Builder
	sb.WriteString(ui.TitleStyle.Render("API Settings") + "\n\n")
	for i, row := range apiSettingsRows {
		isSelected := i == m.apiSettingsCursor
		marker := "  "
		numStr := fmt.Sprintf("[%d]", i+1)
		if isSelected {
			marker = ui.ActiveFlagStyle.Render("▶ ")
			numStr = ui.TitleStyle.Render(numStr)
		} else {
			numStr = ui.SystemMsgStyle.Render(numStr)
		}
		var valStr string
		if row.key == m.apiSettingsEditing {
			valStr = ui.ActiveFlagStyle.Render(m.apiSettingsInput + "_")
		} else {
			var val string
			switch row.key {
			case "api_key":
				if m.settings.NIMAPIKey != "" {
					val = "••••" + m.settings.NIMAPIKey[max(0, len(m.settings.NIMAPIKey)-4):]
				} else {
					val = "(not set)"
				}
			case "base_url":
				val = m.settings.NIMBaseURL
			case "hf_token":
				if m.settings.HFToken != "" {
					val = "••••" + m.settings.HFToken[max(0, len(m.settings.HFToken)-4):]
				} else {
					val = "(not set)"
				}
			}
			if isSelected {
				valStr = lipgloss.NewStyle().Foreground(ui.ColorWhite).Render(val)
			} else {
				valStr = ui.SystemMsgStyle.Render(val)
			}
		}
		sb.WriteString(fmt.Sprintf("%s%s %-14s  %s\n", marker, numStr, row.label, valStr))
	}
	sb.WriteString("\n")
	if m.apiSettingsEditing != "" {
		sb.WriteString(ui.HelpStyle.Render("Enter to save · ESC to cancel"))
	} else {
		sb.WriteString(ui.HelpStyle.Render("↑↓: navigate · Enter: edit · ESC: back"))
	}
	return lipgloss.NewStyle().Height(m.viewportHeight()).Render(
		ui.BorderStyle.Width(m.width - 4).Render(sb.String()),
	)
}

// ─── Interface settings ───────────────────────────────────────────────────────

func (m *AppModel) handleInterfaceSettingsKey(msg tea.KeyMsg) []tea.Cmd {
	m.acConsumedKey = true
	switch msg.Type {
	case tea.KeyEsc:
		m.screen = ScreenSettingsMenu
	case tea.KeyUp:
		if m.ifaceSettingsCursor > 0 {
			m.ifaceSettingsCursor--
		}
	case tea.KeyDown:
		if m.ifaceSettingsCursor < len(interfaceSettingsRows)-1 {
			m.ifaceSettingsCursor++
		}
	case tea.KeyEnter, tea.KeySpace:
		row := interfaceSettingsRows[m.ifaceSettingsCursor]
		switch row.key {
		case "debug":
			m.tuiState.Debug = !m.tuiState.Debug
			m.saveState()
		case "sources":
			m.tuiState.ShowSources = !m.tuiState.ShowSources
			m.saveState()
		case "agent":
			m.tuiState.AgentMode = !m.tuiState.AgentMode
			m.saveState()
		case "confident":
			m.tuiState.Confident = !m.tuiState.Confident
			m.saveState()
		case "rag_only":
			m.tuiState.RAGOnly = !m.tuiState.RAGOnly
			m.saveState()
		}
	}
	return nil
}

func (m AppModel) viewInterfaceSettingsContent() string {
	var sb strings.Builder
	sb.WriteString(ui.TitleStyle.Render("Interface Settings") + "\n\n")
	boolVal := func(b bool) string {
		if b {
			return ui.ActiveFlagStyle.Render("on ")
		}
		return ui.SystemMsgStyle.Render("off")
	}
	for i, row := range interfaceSettingsRows {
		isSelected := i == m.ifaceSettingsCursor
		marker := "  "
		numStr := fmt.Sprintf("[%d]", i+1)
		if isSelected {
			marker = ui.ActiveFlagStyle.Render("▶ ")
			numStr = ui.TitleStyle.Render(numStr)
		} else {
			numStr = ui.SystemMsgStyle.Render(numStr)
		}
		var val bool
		switch row.key {
		case "debug":
			val = m.tuiState.Debug
		case "sources":
			val = m.tuiState.ShowSources
		case "agent":
			val = m.tuiState.AgentMode
		case "confident":
			val = m.tuiState.Confident
		case "rag_only":
			val = m.tuiState.RAGOnly
		}
		label := row.label
		if isSelected {
			label = lipgloss.NewStyle().Foreground(ui.ColorWhite).Render(label)
		} else {
			label = ui.SystemMsgStyle.Render(label)
		}
		sb.WriteString(fmt.Sprintf("%s%s %-18s  %s\n", marker, numStr, label, boolVal(val)))
	}
	sb.WriteString("\n")
	sb.WriteString(ui.HelpStyle.Render("↑↓: navigate · Enter/Space: toggle · ESC: back"))
	return lipgloss.NewStyle().Height(m.viewportHeight()).Render(
		ui.BorderStyle.Width(m.width - 4).Render(sb.String()),
	)
}

// ─── Pipeline screen ──────────────────────────────────────────────────────────

// startIngestFromConfig transitions from config screen to active ingest.
func (m *AppModel) startIngestFromConfig() {
	m.pipelineIngestConfig = false
	m.screen = ScreenChat
	m.ingestInProgress = true
	m.ingestLog = nil // clear log for fresh run
	m.ingestProgress = 0
	m.ingestMessage = ""
	m.viewport.Height = m.viewportHeight()       // shrink viewport to make room for ingest bar
	m.ingestLogVP.Height = m.ingestLogVpHeight() // also resize log viewer for ingest bar
}

// ingestPresets defines the quick-pick options shown in the ingest config screen.
var ingestPresets = []struct {
	label string
	limit int // 0 = all
}{
	{"All messages", 0},
	{"Last 100,000", 100000},
	{"Last 50,000", 50000},
	{"Last 10,000", 10000},
	{"Last 5,000", 5000},
	{"Last 1,000", 1000},
	{"Custom count…", -1},   // opens text input
	{"After date…", -2},     // opens date text input
}

func (m *AppModel) handlePipelineKey(msg tea.KeyMsg) []tea.Cmd {
	m.acConsumedKey = true

	// ── ingest config screen ──────────────────────────────────────────────
	if m.pipelineIngestConfig {
		// Sub-mode: text input for custom count or date
		if m.pipelineConfigCustom {
			switch msg.Type {
			case tea.KeyEsc:
				m.pipelineConfigCustom = false
				m.pipelineInput = ""
			case tea.KeyEnter:
				input := strings.TrimSpace(m.pipelineInput)
				m.pipelineConfigCustom = false
				m.pipelineInput = ""
				preset := ingestPresets[m.pipelineConfigCursor]
				if preset.limit == -1 {
					// custom count
					if n, err := fmt.Sscanf(input, "%d", &m.pipelineConfigLimit); n == 0 || err != nil {
						m.pipelineConfigLimit = 0
					}
				} else {
					// after date
					m.pipelineConfigAfter = input
				}
				// Re-show config screen for confirmation
			case tea.KeyBackspace, tea.KeyDelete:
				if len(m.pipelineInput) > 0 {
					runes := []rune(m.pipelineInput)
					m.pipelineInput = string(runes[:len(runes)-1])
				}
			default:
				if msg.Type == tea.KeyRunes {
					m.pipelineInput += msg.String()
				}
			}
			return nil
		}
		// Main config navigation
		switch msg.Type {
		case tea.KeyEsc:
			// Back to file picker
			m.pipelineIngestConfig = false
			m.pipelineFilePicking = true
		case tea.KeyUp:
			if m.pipelineConfigCursor > 0 {
				m.pipelineConfigCursor--
			}
		case tea.KeyDown:
			if m.pipelineConfigCursor < len(ingestPresets)-1 {
				m.pipelineConfigCursor++
			}
		case tea.KeyEnter:
			preset := ingestPresets[m.pipelineConfigCursor]
			if preset.limit == -1 || preset.limit == -2 {
				// Open text input
				m.pipelineConfigCustom = true
				m.pipelineInput = ""
			} else {
				// Start ingest with chosen settings
				m.startIngestFromConfig()
				return []tea.Cmd{m.spinner.Tick, pipelineIngestCmd(m.bridge, m.pipelineConfigFile, m.pipelineConfigLimit, m.pipelineConfigAfter)}
			}
		case tea.KeyRunes:
			// 'g' = go / start ingest with current settings
			if msg.String() == "g" {
				m.startIngestFromConfig()
				return []tea.Cmd{m.spinner.Tick, pipelineIngestCmd(m.bridge, m.pipelineConfigFile, m.pipelineConfigLimit, m.pipelineConfigAfter)}
			}
		}
		return nil
	}

	// ── file picker mode (ingest) ─────────────────────────────────────────
	if m.pipelineFilePicking {
		switch msg.Type {
		case tea.KeyEsc:
			m.pipelineFilePicking = false
		case tea.KeyUp:
			if m.pipelineFileCursor > 0 {
				m.pipelineFileCursor--
			}
		case tea.KeyDown:
			if m.pipelineFileCursor < len(m.chatFileList)-1 {
				m.pipelineFileCursor++
			}
		case tea.KeyEnter:
			if len(m.chatFileList) > 0 {
				filename := m.chatFileList[m.pipelineFileCursor]
				path := filepath.Join(m.ragDir, "raw", filename)
				// Open config screen instead of ingesting immediately.
				m.pipelineFilePicking = false
				m.pipelineIngestConfig = true
				m.pipelineConfigFile = path
				m.pipelineConfigCursor = 0
				m.pipelineConfigLimit = 0
				m.pipelineConfigAfter = ""
				m.pipelineConfigCustom = false
				m.ingestFilename = filename
				m.ingestNewSlug = deriveSlugFromFilename(filename)
			}
		}
		return nil
	}

	// ── text input mode (test query) ──────────────────────────────────────
	if m.pipelineInputMode {
		switch msg.Type {
		case tea.KeyEsc:
			m.pipelineInputMode = false
			m.pipelineInput = ""
		case tea.KeyEnter:
			input := strings.TrimSpace(m.pipelineInput)
			m.pipelineInputMode = false
			m.pipelineInput = ""
			action := pipelineMenuItems[m.pipelineCursor].action
			switch action {
			case "test":
				return []tea.Cmd{pipelineTestCmd(m.bridge, input, m.settings.FinalK, m.tuiState.Window)}
			}
		case tea.KeyBackspace, tea.KeyDelete:
			if len(m.pipelineInput) > 0 {
				runes := []rune(m.pipelineInput)
				m.pipelineInput = string(runes[:len(runes)-1])
			}
		default:
			if msg.Type == tea.KeyRunes {
				m.pipelineInput += msg.String()
			}
		}
		return nil
	}

	// ── menu navigation ───────────────────────────────────────────────────
	switch msg.Type {
	case tea.KeyEsc:
		m.screen = ScreenSettingsMenu
	case tea.KeyUp:
		if m.pipelineCursor > 0 {
			m.pipelineCursor--
		}
	case tea.KeyDown:
		if m.pipelineCursor < len(pipelineMenuItems)-1 {
			m.pipelineCursor++
		}
	case tea.KeyEnter:
		item := pipelineMenuItems[m.pipelineCursor]
		switch item.action {
		case "rebuild":
			return []tea.Cmd{pipelineRebuildCmd(m.bridge)}
		case "stats":
			m.screen = ScreenChat
			return []tea.Cmd{fetchStatsCmd(m.bridge)}
		case "check":
			m.screen = ScreenChat
			return []tea.Cmd{fetchCheckCmd(m.bridge)}
		case "test":
			m.pipelineInputMode = true
			m.pipelinePrompt = "Enter query"
			m.pipelineInput = ""
		case "ingest":
			m.pipelineFilePicking = true
			m.pipelineFileCursor = 0
			return []tea.Cmd{loadChatFileListCmd(m.ragDir)}
		}
	}
	return nil
}

func (m AppModel) viewPipelineContent() string {
	accentStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
	dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
	valStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite)
	activeStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)

	var sb strings.Builder
	sb.WriteString(ui.TitleStyle.Render("Pipeline Management") + "\n\n")

	if m.pipelineIngestConfig {
		// ── ingest config screen ──────────────────────────────────────
		stem := strings.TrimSuffix(filepath.Base(m.pipelineConfigFile), filepath.Ext(m.pipelineConfigFile))
		sb.WriteString(accentStyle.Render("◆ Ingest: "+stem) + "\n\n")

		// Show current settings summary
		limitStr := "all messages"
		if m.pipelineConfigLimit > 0 {
			limitStr = fmt.Sprintf("last %s messages", formatCount(m.pipelineConfigLimit))
		}
		afterStr := ""
		if m.pipelineConfigAfter != "" {
			afterStr = "  ·  after " + m.pipelineConfigAfter
		}
		sb.WriteString(dimStyle.Render("  Current: ") + valStyle.Render(limitStr+afterStr) + "\n\n")

		if m.pipelineConfigCustom {
			// text input sub-mode
			preset := ingestPresets[m.pipelineConfigCursor]
			prompt := "Enter count"
			if preset.limit == -2 {
				prompt = "After date (YYYY-MM-DD)"
			}
			sb.WriteString(dimStyle.Render("  "+prompt+": ") + activeStyle.Render(m.pipelineInput+"▌") + "\n\n")
			sb.WriteString(ui.HelpStyle.Render("Enter to confirm · ESC to cancel"))
		} else {
			sb.WriteString(dimStyle.Render("  How much to index:\n\n"))
			for i, p := range ingestPresets {
				isSelected := i == m.pipelineConfigCursor
				marker := "   "
				timeHint := ingestTimeHint(p.limit)
				if isSelected {
					sb.WriteString(activeStyle.Render(" ▶ ") +
						activeStyle.Render(fmt.Sprintf("%-16s", p.label)) +
						dimStyle.Render("  "+timeHint) + "\n")
				} else {
					sb.WriteString(dimStyle.Render(marker+fmt.Sprintf("%-16s", p.label)) +
						dimStyle.Render("  "+timeHint) + "\n")
				}
			}
			sb.WriteString("\n")
			sb.WriteString(ui.HelpStyle.Render("↑↓: navigate · Enter: select · G: start ingest · ESC: back"))
		}

	} else if m.pipelineFilePicking {
		// ── file picker ───────────────────────────────────────────────
		rawPath := filepath.Join(m.ragDir, "raw")
		sb.WriteString(accentStyle.Render("◆ Ingest — pick a file to add") + "\n")
		sb.WriteString(dimStyle.Render("  Drop your file into: ") + valStyle.Render(rawPath) + "\n\n")
		if len(m.chatFileList) == 0 {
			sb.WriteString(dimStyle.Render("  No files found in raw/") + "\n\n")
			sb.WriteString(dimStyle.Render("  Supports: .json (Slack/Discord exports) and .csv (email datasets)"))
		} else {
			for i, f := range m.chatFileList {
				isSelected := i == m.pipelineFileCursor
				ext := strings.ToUpper(strings.TrimPrefix(filepath.Ext(f), "."))
				stem := strings.TrimSuffix(f, filepath.Ext(f))
				extTag := dimStyle.Render("[" + ext + "]")
				if isSelected {
					sb.WriteString(ui.ActiveFlagStyle.Render(" ▶ ") + valStyle.Bold(true).Render(stem) + "  " + extTag + "\n")
				} else {
					sb.WriteString(dimStyle.Render("   "+stem) + "  " + extTag + "\n")
				}
			}
		}
		sb.WriteString("\n")
		sb.WriteString(ui.HelpStyle.Render("↑↓: navigate · Enter: configure & ingest · ESC: back"))

	} else if m.pipelineInputMode {
		// ── text input ────────────────────────────────────────────────
		sb.WriteString(ui.SystemMsgStyle.Render(m.pipelinePrompt+": ") + ui.ActiveFlagStyle.Render(m.pipelineInput+"_"))
		sb.WriteString("\n\n")
		sb.WriteString(ui.HelpStyle.Render("Enter to confirm · ESC to cancel"))

	} else {
		// ── menu ──────────────────────────────────────────────────────
		sb.WriteString(dimStyle.Render("  Got new data? Drop a JSON or CSV file into raw/ → Ingest.") + "\n\n")
		descriptions := map[string]string{
			"ingest":  "add messages from a new file — choose how much to index",
			"check":   "show the date range to use when exporting from Slack/Discord",
			"rebuild": "reprocess all raw data from scratch — use if index is broken or corrupted (slow)",
			"test":    "run a retrieval query without calling the LLM",
			"stats":   "show chunk count and index info",
		}
		for i, item := range pipelineMenuItems {
			isSelected := i == m.pipelineCursor
			if isSelected {
				sb.WriteString(ui.ActiveFlagStyle.Render("▶ ") +
					accentStyle.Render(item.label) + "\n")
				sb.WriteString(dimStyle.Render("    "+descriptions[item.action]) + "\n\n")
			} else {
				sb.WriteString(dimStyle.Render("  "+item.label) + "\n\n")
			}
		}
		sb.WriteString(ui.HelpStyle.Render("↑↓: navigate · Enter: run · ESC: back"))
	}

	return lipgloss.NewStyle().Height(m.viewportHeight()).Render(
		ui.BorderStyle.Width(m.width - 4).Render(sb.String()),
	)
}

// ingestTimeHint returns a rough time estimate string for a given message limit.
func ingestTimeHint(limit int) string {
	switch {
	case limit == 0:
		return "time varies"
	case limit <= 1000:
		return "~1–2 min"
	case limit <= 5000:
		return "~5 min"
	case limit <= 10000:
		return "~10 min"
	case limit <= 50000:
		return "~45 min"
	case limit <= 100000:
		return "~1.5 hrs"
	default:
		return "time varies"
	}
}


func formatCount(n int) string {
	s := fmt.Sprintf("%d", n)
	if n < 1000 {
		return s
	}
	// Insert commas
	var result []byte
	for i, c := range []byte(s) {
		if i > 0 && (len(s)-i)%3 == 0 {
			result = append(result, ',')
		}
		result = append(result, c)
	}
	return string(result)
}

func (m *AppModel) handleSettingsKey(msg tea.KeyMsg) []tea.Cmd {
	m.acConsumedKey = true // settings screen always consumes all keys

	if m.settingsEditing != "" {
		// Text editing mode.
		switch msg.Type {
		case tea.KeyEsc:
			m.settingsEditing = ""
			m.settingsInput = ""
		case tea.KeyEnter:
			m.applySettingsEdit()
			m.settingsEditing = ""
			m.settingsInput = ""
		case tea.KeyBackspace, tea.KeyDelete:
			if len(m.settingsInput) > 0 {
				runes := []rune(m.settingsInput)
				m.settingsInput = string(runes[:len(runes)-1])
			}
		default:
			if msg.Type == tea.KeyRunes {
				m.settingsInput += msg.String()
			}
		}
		return nil
	}

	// Navigation mode — arrow keys + Enter.
	switch msg.Type {
	case tea.KeyEsc:
		m.screen = ScreenSettingsMenu
		m.settingsCursor = 0
	case tea.KeyUp:
		if m.settingsCursor > 0 {
			m.settingsCursor--
		}
	case tea.KeyDown:
		if m.settingsCursor < len(settingsRows)-1 {
			m.settingsCursor++
		}
	case tea.KeyEnter:
		return m.activateSettingsRow()
	}
	return nil
}

// activateSettingsRow handles Enter on the currently highlighted settings row.
func (m *AppModel) activateSettingsRow() []tea.Cmd {
	if m.settingsCursor >= len(settingsRows) {
		return nil
	}
	row := settingsRows[m.settingsCursor]
	switch row.key {
	case "model":
		// Open the model picker.
		m.screen = ScreenModelPicker
		m.modelCustomMode = false
		m.modelCustomInput = ""
		m.modelCursor = 0
		for i, mdl := range allModels {
			if mdl == m.settings.NIMModel {
				m.modelCursor = i
				break
			}
		}
	case "thinking":
		// Toggle immediately.
		m.settings.ThinkingMode = !m.settings.ThinkingMode
		_ = m.settings.Save()
	case "rag_only":
		// Toggle immediately.
		m.tuiState.RAGOnly = !m.tuiState.RAGOnly
		m.saveState()
	case "nick":
		// Open nickname editor.
		m.screen = ScreenNick
		m.nickStep = 0
		m.nickModelIdx = 0
		m.nickInput = ""
	default:
		// Enter text editing mode.
		m.settingsEditing = row.key
		switch row.key {
		case "temperature":
			m.settingsInput = fmt.Sprintf("%g", m.settings.Temperature)
		case "top_p":
			m.settingsInput = fmt.Sprintf("%g", m.settings.TopP)
		case "max_tokens":
			m.settingsInput = strconv.Itoa(m.settings.MaxTokens)
		case "final_k":
			m.settingsInput = strconv.Itoa(m.settings.FinalK)
		case "window":
			m.settingsInput = strconv.Itoa(m.tuiState.Window)
		case "api_key":
			m.settingsInput = ""
		case "base_url":
			m.settingsInput = m.settings.NIMBaseURL
		}
	}
	return nil
}

func (m *AppModel) handleChatsKey(msg tea.KeyMsg) []tea.Cmd {
	m.acConsumedKey = true // chats screen always consumes all keys

	switch msg.Type {
	case tea.KeyEsc:
		m.screen = ScreenChat
	case tea.KeyUp:
		if m.chatCursor > 0 {
			m.chatCursor--
		}
	case tea.KeyDown:
		if m.chatCursor < len(m.chatSlugList)-1 {
			m.chatCursor++
		}
	case tea.KeyEnter:
		if m.chatCursor >= 0 && m.chatCursor < len(m.chatSlugList) {
			slug := m.chatSlugList[m.chatCursor]
			m.screen = ScreenChat
			return []tea.Cmd{switchChatCmd(m.bridge, slug)}
		}
	default:
		s := msg.String()
		if s == "q" {
			m.screen = ScreenChat
			return nil
		}
		// Number shortcut.
		idx, err := strconv.Atoi(s)
		if err == nil && idx >= 1 && idx <= len(m.chatSlugList) {
			slug := m.chatSlugList[idx-1]
			m.screen = ScreenChat
			return []tea.Cmd{switchChatCmd(m.bridge, slug)}
		}
	}
	return nil
}

func (m *AppModel) handleHelpKey(msg tea.KeyMsg) []tea.Cmd {
	if m.helpSection == "" {
		switch msg.Type {
		case tea.KeyEsc:
			m.screen = ScreenChat
			m.acConsumedKey = true
		case tea.KeyEnter:
			m.helpSection = helpSections[m.helpCursor].id
			m.refreshHelpContent()
			m.helpVp.GotoTop()
			m.acConsumedKey = true
		case tea.KeyUp:
			if m.helpCursor > 0 {
				m.helpCursor--
				m.refreshHelpContent()
			}
			m.acConsumedKey = true
		case tea.KeyDown:
			if m.helpCursor < len(helpSections)-1 {
				m.helpCursor++
				m.refreshHelpContent()
			}
			m.acConsumedKey = true
		}
		return nil
	}

	switch msg.Type {
	case tea.KeyEsc:
		m.helpSection = ""
		m.refreshHelpContent()
		m.acConsumedKey = true
	case tea.KeyBackspace, tea.KeyLeft:
		m.helpSection = ""
		m.refreshHelpContent()
		m.acConsumedKey = true
	case tea.KeyUp:
		m.helpVp.LineUp(1)
		m.acConsumedKey = true
	case tea.KeyDown:
		m.helpVp.LineDown(1)
		m.acConsumedKey = true
	case tea.KeyPgUp:
		m.helpVp.HalfViewUp()
		m.acConsumedKey = true
	case tea.KeyPgDown:
		m.helpVp.HalfViewDown()
		m.acConsumedKey = true
	}
	// All other keys (letters etc.) fall through to the textarea below.
	return nil
}

func (m *AppModel) handleModelPickerKey(msg tea.KeyMsg) []tea.Cmd {
	m.acConsumedKey = true

	if m.modelCustomMode {
		// Custom model text input.
		switch msg.Type {
		case tea.KeyEsc:
			m.modelCustomMode = false
			m.modelCustomInput = ""
		case tea.KeyEnter:
			if v := strings.TrimSpace(m.modelCustomInput); v != "" {
				m.settings.NIMModel = v
				_ = m.settings.Save()
			}
			m.modelCustomMode = false
			m.modelCustomInput = ""
			m.screen = ScreenSettings
		case tea.KeyBackspace, tea.KeyDelete:
			if len(m.modelCustomInput) > 0 {
				runes := []rune(m.modelCustomInput)
				m.modelCustomInput = string(runes[:len(runes)-1])
			}
		default:
			if msg.Type == tea.KeyRunes {
				m.modelCustomInput += msg.String()
			}
		}
		return nil
	}

	// List navigation. Indices 0..len(allModels)-1 = presets, len = Custom.
	maxIdx := len(allModels)
	switch msg.Type {
	case tea.KeyEsc:
		m.screen = ScreenSettings
	case tea.KeyUp:
		if m.modelCursor > 0 {
			m.modelCursor--
		}
	case tea.KeyDown:
		if m.modelCursor < maxIdx {
			m.modelCursor++
		}
	case tea.KeyEnter:
		if m.modelCursor < len(allModels) {
			m.settings.NIMModel = allModels[m.modelCursor]
			_ = m.settings.Save()
			m.screen = ScreenSettings
		} else {
			// Custom option selected.
			m.modelCustomMode = true
			m.modelCustomInput = ""
		}
	}
	return nil
}

// ─── Autocomplete ─────────────────────────────────────────────────────────────

func (m *AppModel) applySettingsEdit() {
	v := strings.TrimSpace(m.settingsInput)
	if v == "" {
		return
	}
	switch m.settingsEditing {
	case "temperature":
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			m.settings.Temperature = f
		}
	case "top_p":
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			m.settings.TopP = f
		}
	case "max_tokens":
		if n, err := strconv.Atoi(v); err == nil {
			m.settings.MaxTokens = n
		}
	case "final_k":
		if n, err := strconv.Atoi(v); err == nil {
			m.settings.FinalK = n
		}
	case "window":
		if n, err := strconv.Atoi(v); err == nil {
			m.tuiState.Window = n
			m.saveState()
		}
	case "api_key":
		m.settings.NIMAPIKey = v
	case "base_url":
		m.settings.NIMBaseURL = v
	}
	_ = m.settings.Save()
}

// updateAutocomplete recomputes suggestions based on current textarea content.
// It also refreshes m.viewport.Height so the viewport shrinks to make room.
func (m *AppModel) updateAutocomplete() {
	text := m.textarea.Value()
	// Activate on "/" alone or "/prefix" (no spaces = still typing command name).
	if !strings.HasPrefix(text, "/") || len(text) < 1 || strings.ContainsRune(text, ' ') {
		m.acActive = false
		m.acSuggestions = nil
		m.acSelected = 0
		m.viewport.Height = m.viewportHeight()
		return
	}

	prefix := strings.ToLower(text)
	var matches []cmdSuggestion
	for _, c := range allCommands {
		if strings.HasPrefix(c.cmd, prefix) {
			matches = append(matches, c)
		}
	}

	if len(matches) == 0 {
		m.acActive = false
		m.acSuggestions = nil
		m.acSelected = 0
		m.viewport.Height = m.viewportHeight()
		return
	}

	// Find the highest-priority (lowest prio number) match → goes first.
	bestIdx := 0
	for i, c := range matches {
		if c.prio < matches[bestIdx].prio {
			bestIdx = i
		}
	}
	best := matches[bestIdx]
	rest := make([]cmdSuggestion, 0, len(matches)-1)
	for i, c := range matches {
		if i != bestIdx {
			rest = append(rest, c)
		}
	}
	sort.Slice(rest, func(i, j int) bool { return rest[i].cmd < rest[j].cmd })

	suggestions := append([]cmdSuggestion{best}, rest...)
	if len(suggestions) > 5 {
		suggestions = suggestions[:5]
	}

	m.acSuggestions = suggestions
	m.acActive = true
	if m.acSelected >= len(m.acSuggestions) {
		m.acSelected = 0
	}
	m.viewport.Height = m.viewportHeight()
}

// renderAutocomplete renders the suggestion list below the input box.
func (m AppModel) renderAutocomplete() string {
	if !m.acActive || len(m.acSuggestions) == 0 {
		return ""
	}

	typed := m.textarea.Value() // e.g. "/ex"
	matchStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
	restStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite)
	descStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
	selectedStyle := lipgloss.NewStyle().Foreground(ui.ColorGreen).Bold(true)
	firstStyle := lipgloss.NewStyle().Foreground(ui.ColorYellow).Bold(true)

	var lines []string
	for i, s := range m.acSuggestions {
		// Color the typed prefix throughout; keep rest of cmd in white.
		var cmdRendered string
		if len(s.cmd) >= len(typed) {
			matchPart := matchStyle.Render(s.cmd[:len(typed)])
			restPart := restStyle.Render(s.cmd[len(typed):])
			cmdRendered = matchPart + restPart
		} else {
			cmdRendered = matchStyle.Render(s.cmd)
		}
		desc := descStyle.Render("  " + s.desc)

		switch {
		case i == m.acSelected && i == 0:
			// Selected AND first: green bold for the whole thing.
			cmdRendered = selectedStyle.Render("▶ "+s.cmd[:len(typed)]) +
				selectedStyle.Render(s.cmd[len(typed):])
		case i == m.acSelected:
			cmdRendered = selectedStyle.Render("▶ ") + cmdRendered
		case i == 0:
			// First/most-common but not selected: yellow bold to stand out.
			if len(s.cmd) >= len(typed) {
				cmdRendered = "  " + firstStyle.Render(s.cmd[:len(typed)]) +
					firstStyle.Render(s.cmd[len(typed):])
			} else {
				cmdRendered = "  " + firstStyle.Render(s.cmd)
			}
		default:
			cmdRendered = "  " + cmdRendered
		}

		lines = append(lines, cmdRendered+desc)
	}

	hint := descStyle.Render("  tab: complete · ↑↓: cycle · esc: clear")
	lines = append(lines, hint)
	return strings.Join(lines, "\n")
}

// ─── Inline commands ──────────────────────────────────────────────────────────

func (m *AppModel) handleInlineCmd(raw string) []tea.Cmd {
	parts := strings.Fields(raw)
	if len(parts) == 0 {
		return nil
	}
	cmdName := strings.ToLower(parts[0])

	switch cmdName {
	case "/debug":
		m.tuiState.Debug = !m.tuiState.Debug
		m.saveState()
		m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("debug: %v", m.tuiState.Debug)})
		if m.tuiState.Debug {
			m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("log file: %s", filepath.Join(m.ragDir, "ragtag.log"))})
		}
		workers.Logf(m.ragDir, "cmd /debug -> %v", m.tuiState.Debug)

	case "/sources":
		m.tuiState.ShowSources = !m.tuiState.ShowSources
		m.saveState()
		m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("sources: %v", m.tuiState.ShowSources)})

	case "/confident":
		m.tuiState.Confident = !m.tuiState.Confident
		m.saveState()
		m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("confident: %v", m.tuiState.Confident)})

	case "/thinking":
		m.settings.ThinkingMode = !m.settings.ThinkingMode
		_ = m.settings.Save()
		m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("thinking: %v", m.settings.ThinkingMode)})

	case "/rag":
		m.tuiState.RAGOnly = !m.tuiState.RAGOnly
		m.saveState()
		m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("rag-only: %v", m.tuiState.RAGOnly)})
		m.maybeWarnMissingAPIKey()

	case "/mode":
		modes := []string{"plain", "structured", "rich"}
		if len(parts) > 1 {
			arg := strings.ToLower(strings.TrimSpace(parts[1]))
			valid := false
			for _, m2 := range modes {
				if arg == m2 {
					valid = true
					break
				}
			}
			if valid {
				m.tuiState.OutputMode = arg
				m.saveState()
				modeDesc := map[string]string{
					"plain":      "plain prose (default)",
					"structured": "structured JSON with summary + key points",
					"rich":       "rich components — chat logs, tables, timelines",
				}
				m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("output mode → %s  (%s)", arg, modeDesc[arg])})
			} else {
				// Cycle through modes if no arg given
				current := m.tuiState.OutputMode
				if current == "" {
					current = "plain"
				}
				next := "plain"
				for i, mo := range modes {
					if mo == current {
						next = modes[(i+1)%len(modes)]
						break
					}
				}
				m.tuiState.OutputMode = next
				m.saveState()
				m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("output mode → %s", next)})
			}
		} else {
			// No arg: cycle
			current := m.tuiState.OutputMode
			if current == "" {
				current = "plain"
			}
			next := "plain"
			for i, mo := range modes {
				if mo == current {
					next = modes[(i+1)%len(modes)]
					break
				}
			}
			m.tuiState.OutputMode = next
			m.saveState()
			m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("output mode → %s", next)})
		}

	case "/window":
		if len(parts) > 1 {
			if n, err := strconv.Atoi(parts[1]); err == nil {
				m.tuiState.Window = n
				m.saveState()
				m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("window: %d", n)})
			} else {
				m.addMessage(ChatMessage{Role: "error", Content: "Usage: /window N"})
			}
		}

	case "/k":
		if len(parts) > 1 {
			if n, err := strconv.Atoi(parts[1]); err == nil {
				m.settings.FinalK = n
				_ = m.settings.Save()
				m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("final_k: %d", n)})
			} else {
				m.addMessage(ChatMessage{Role: "error", Content: "Usage: /k N"})
			}
		}

	case "/chats":
		m.screen = ScreenChats
		// Pre-position cursor on the active chat.
		m.chatCursor = 0
		for i, slug := range m.chatSlugList {
			if slug == m.activeChat {
				m.chatCursor = i
				break
			}
		}
		return []tea.Cmd{fetchChatListCmd(m.bridge)}

	case "/settings":
		m.screen = ScreenSettingsMenu
		m.settingsMenuCursor = 0

	case "/model":
		if len(parts) > 1 {
			arg := strings.Join(parts[1:], " ")
			if resolved, ok := m.resolveModel(arg); ok {
				m.settings.NIMModel = resolved
				_ = m.settings.Save()
				m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("model: %s", resolved)})
			} else {
				var sb strings.Builder
				sb.WriteString(fmt.Sprintf("Unknown model %q. Available:\n", arg))
				for i, mdl := range allModels {
					nicks := m.nicknamesFor(mdl)
					nickStr := ""
					if len(nicks) > 0 {
						nickStr = "  (" + strings.Join(nicks, ", ") + ")"
					}
					sb.WriteString(fmt.Sprintf("  [%2d] %s%s\n", i+1, mdl, nickStr))
				}
				m.addMessage(ChatMessage{Role: "system", Content: sb.String()})
			}
		} else {
			// Open model picker.
			m.screen = ScreenModelPicker
			m.modelCustomMode = false
			m.modelCustomInput = ""
			m.modelCursor = 0
			for i, mdl := range allModels {
				if mdl == m.settings.NIMModel {
					m.modelCursor = i
					break
				}
			}
		}

	case "/pause":
		m.pauseCurrentOp()

	case "/update":
		if m.updateAvailable == "" {
			// No cached update — do a fresh check, then update if newer.
			m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("Checking for updates (current: %s)…", AppVersion)})
			return []tea.Cmd{checkVersionThenUpdateCmd()}
		}
		m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("Downloading %s…", m.updateAvailable)})
		return []tea.Cmd{selfUpdateCmd()}

	case "/pipeline":
		m.screen = ScreenPipeline
		m.pipelineCursor = 0
		m.pipelineInputMode = false
		m.pipelineInput = ""
		m.pipelineFilePicking = false

	case "/ingest":
		if m.ingestInProgress {
			m.refreshIngestLog()
			m.screen = ScreenIngestLog
		} else {
			m.screen = ScreenPipeline
			m.pipelineCursor = 0
			m.pipelineFilePicking = true
			m.pipelineFileCursor = 0
			return []tea.Cmd{loadChatFileListCmd(m.ragDir)}
		}

	case "/check":
		return []tea.Cmd{fetchCheckCmd(m.bridge)}

	case "/view":
		m.screen = ScreenChatList
		m.chatFileCursor = 0
		return []tea.Cmd{loadChatFileListCmd(m.ragDir)}

	case "/minresults":
		if len(parts) > 1 {
			if n, err := strconv.Atoi(parts[1]); err == nil && n >= 0 {
				m.tuiState.MinResults = n
				m.saveState()
				m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("min_results: %d", n)})
			} else {
				m.addMessage(ChatMessage{Role: "error", Content: "Usage: /minresults N  (N >= 0)"})
			}
		} else {
			m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("min_results: %d", m.tuiState.MinResults)})
		}

	case "/threshold":
		if len(parts) > 1 {
			if f, err := strconv.ParseFloat(parts[1], 64); err == nil && f >= 0 {
				m.tuiState.ScoreThreshold = f
				m.saveState()
				m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("score_threshold: %.3f", f)})
			} else {
				m.addMessage(ChatMessage{Role: "error", Content: "Usage: /threshold F  (F >= 0.0)"})
			}
		} else {
			m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("score_threshold: %.3f", m.tuiState.ScoreThreshold)})
		}

	case "/help":
		m.screen = ScreenHelp
		m.helpSection = ""
		m.helpCursor = 0
		m.helpVp.Width = m.helpVpWidth()
		m.helpVp.Height = m.helpVpHeight()
		m.refreshHelpContent()
		m.helpVp.GotoTop()

	case "/stats":
		return []tea.Cmd{fetchStatsCmd(m.bridge)}

	case "/back", "/exit":
		if m.screen != ScreenChat {
			m.screen = ScreenChat
		}

	case "/agent":
		rest := strings.TrimSpace(raw[len("/agent"):])
		if rest == "" {
			m.tuiState.AgentMode = !m.tuiState.AgentMode
			m.saveState()
			workers.Logf(m.ragDir, "cmd /agent toggle -> %v", m.tuiState.AgentMode)
			return nil
		}
		restParts := strings.Fields(rest)
		maxSteps := -1
		agentQuery := rest
		if len(restParts) > 1 {
			if n, err := strconv.Atoi(restParts[0]); err == nil {
				maxSteps = n
				agentQuery = strings.Join(restParts[1:], " ")
			}
		}
		if agentQuery != "" {
			return m.startAgentFlow(agentQuery, maxSteps)
		}

	case "/tools":
		if len(parts) == 1 {
			m.addMessage(ChatMessage{Role: "system", Content: m.agentToolsSummary()})
			workers.Logf(m.ragDir, "cmd /tools show")
			return nil
		}
		if len(parts) != 3 {
			m.addMessage(ChatMessage{Role: "error", Content: "Usage: /tools  OR  /tools <search|context|all> <on|off>"})
			return nil
		}
		target := strings.ToLower(strings.TrimSpace(parts[1]))
		state := strings.ToLower(strings.TrimSpace(parts[2]))
		enabled, ok := parseToolState(state)
		if !ok {
			m.addMessage(ChatMessage{Role: "error", Content: "Usage: /tools <search|context|all> <on|off>"})
			return nil
		}
		switch target {
		case "search", "search_rag":
			m.tuiState.AgentToolSearch = enabled
		case "context", "expand_context", "window":
			m.tuiState.AgentToolContext = enabled
		case "all":
			m.tuiState.AgentToolSearch = enabled
			m.tuiState.AgentToolContext = enabled
		default:
			m.addMessage(ChatMessage{Role: "error", Content: "Unknown tool. Use search, context, or all."})
			return nil
		}
		m.saveState()
		m.addMessage(ChatMessage{Role: "system", Content: m.agentToolsSummary()})
		workers.Logf(m.ragDir, "cmd /tools target=%s enabled=%v", target, enabled)
		return nil

	case "/clear":
		m.messages = nil
		m.renderMessages()

	default:
		m.addMessage(ChatMessage{Role: "system", Content: "unknown command. type /help for help."})
	}
	return nil
}

func parseToolState(s string) (bool, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "on", "true", "enable", "enabled":
		return true, true
	case "off", "false", "disable", "disabled":
		return false, true
	default:
		return false, false
	}
}

func onOffLabel(v bool) string {
	if v {
		return "on"
	}
	return "off"
}

func (m AppModel) agentToolsSummary() string {
	return fmt.Sprintf(
		"agent tools\n  search   = %s\n  context  = %s\nusage: /tools <search|context|all> <on|off>",
		onOffLabel(m.tuiState.AgentToolSearch),
		onOffLabel(m.tuiState.AgentToolContext),
	)
}

func (m AppModel) agentToolWindow() int {
	if m.tuiState.AgentToolContext {
		return m.tuiState.Window
	}
	return 0
}

// ─── Flow starters ────────────────────────────────────────────────────────────

// pauseCurrentOp cancels the active AI operation (streaming, retrieval, agent).
func (m *AppModel) pauseCurrentOp() {
	if m.cancelFn != nil {
		m.cancelFn()
		m.cancelFn = nil
	}
	// Finalise any partial streaming message.
	if m.streamBuf.Len() > 0 {
		content := m.streamBuf.String() + " *(paused)*"
		m.streamBuf.Reset()
		if len(m.messages) > 0 && m.messages[len(m.messages)-1].Role == "assistant" {
			m.messages[len(m.messages)-1].Content = content
			m.messages[len(m.messages)-1].Streaming = false
		} else {
			m.messages = append(m.messages, ChatMessage{Role: "assistant", Content: content})
		}
		m.renderMessages()
	}
	m.appState = StateIdle
	m.addMessage(ChatMessage{Role: "system", Content: "operation paused"})
}

func (m *AppModel) startRetrievalFlow(query string) []tea.Cmd {
	if !m.bridgeReady {
		m.addMessage(ChatMessage{Role: "error", Content: "Retriever not ready yet. Please wait."})
		return nil
	}
	if m.cancelFn != nil {
		m.cancelFn()
	}
	m.activeCtx, m.cancelFn = context.WithCancel(context.Background())
	m.appState = StateRetrieving
	m.addMessage(ChatMessage{Role: "user", Content: query})
	return []tea.Cmd{
		m.spinner.Tick,
		retrieveCmd(m.bridge, query, m.settings.FinalK, m.tuiState.Window, m.tuiState.Debug, m.tuiState.MinResults, m.tuiState.ScoreThreshold),
	}
}

func (m *AppModel) startAgentFlow(question string, maxSteps int) []tea.Cmd {
	if !m.bridgeReady {
		m.addMessage(ChatMessage{Role: "error", Content: "Retriever not ready yet. Please wait."})
		workers.Logf(m.ragDir, "agent start blocked: bridge not ready question=%q", question)
		return nil
	}
	if !m.tuiState.AgentToolSearch {
		m.addMessage(ChatMessage{Role: "error", Content: "agent search tool is off. use /tools search on"})
		workers.Logf(m.ragDir, "agent start blocked: search tool disabled question=%q", question)
		return nil
	}
	m.appState = StateAgentStep
	m.agentQuestion = question
	m.agentMaxSteps = maxSteps
	m.agentStep = 0
	m.agentSearches = nil
	m.agentFailed = nil
	m.agentCtx.Reset()
	m.agentConsecFail = 0

	label := "unlimited steps"
	if maxSteps > 0 {
		label = fmt.Sprintf("max %d steps", maxSteps)
	}
	m.addMessage(ChatMessage{Role: "user", Content: question})
	m.addMessage(ChatMessage{Role: "agentic_step", Content: fmt.Sprintf("[agentic mode · %s]", label)})
	workers.Logf(
		m.ragDir,
		"agent start question=%q maxSteps=%d tools={search:%v context:%v} window=%d debug=%v model=%s",
		question,
		maxSteps,
		m.tuiState.AgentToolSearch,
		m.tuiState.AgentToolContext,
		m.agentToolWindow(),
		m.tuiState.Debug,
		m.settings.NIMModel,
	)

	return []tea.Cmd{m.spinner.Tick, m.runAgentStepCmd()}
}

// ─── Agent helpers ────────────────────────────────────────────────────────────

func (m *AppModel) agentHardCap() int {
	if m.agentMaxSteps > 0 {
		return m.agentMaxSteps
	}
	if m.tuiState.Confident {
		return 40
	}
	return 20
}

// runAgentStepCmd captures all state needed for this step as local variables
// (so the closure is safe to run in a goroutine).
func (m *AppModel) runAgentStepCmd() tea.Cmd {
	question := m.agentQuestion
	searches := make([]string, len(m.agentSearches))
	copy(searches, m.agentSearches)
	failed := make([]string, len(m.agentFailed))
	copy(failed, m.agentFailed)
	accumulated := m.agentCtx.String()
	maxSteps := m.agentMaxSteps
	confident := m.tuiState.Confident
	contextEnabled := m.tuiState.AgentToolContext
	cfg := m.settings.ToLLMConfig()
	cfg.LogDir = m.ragDir

	return func() (msg tea.Msg) {
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		defer func() {
			if r := recover(); r != nil {
				crashLog("panic in runAgentStepCmd: %v\n%s", r, debug2.Stack())
				msg = AgentLLMDoneMsg{Action: "ANSWER", Payload: fmt.Sprintf("[agent panic: %v]", r)}
			}
		}()
		system := buildAgentSystemPrompt(confident, maxSteps, len(searches), contextEnabled)

		var remaining string
		if maxSteps > 0 {
			remaining = fmt.Sprintf("Searches remaining: %d", maxSteps-len(searches))
		} else {
			remaining = fmt.Sprintf("Searches so far: %d", len(searches))
		}

		ctxText := accumulated
		if ctxText == "" {
			ctxText = "(none yet)"
		}

		prompt := fmt.Sprintf(
			"Question: %s\n\nSearches done: %v\nLow-quality searches (avoid similar): %v\n%s\n\nIMPORTANT: Do NOT append words to previous queries. Think of a completely new angle.\nContext retrieved so far:\n%s\n\nWhat do you do next? Reply with exactly one SEARCH: ... line or one ANSWER: ... response.",
			question, searches, failed, remaining, ctxText,
		)
		workers.Logf(
			m.ragDir,
			"agent llm request step=%d searches=%d failed=%d contextChars=%d tools={context:%v}",
			len(searches)+1,
			len(searches),
			len(failed),
			len(ctxText),
			contextEnabled,
		)

		messages := []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: system},
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		}

		resp, err := workers.CallLLM(ctx, cfg, messages)
		if err != nil {
			return AgentLLMDoneMsg{Action: "ANSWER", Payload: fmt.Sprintf("[LLM error: %v]", err), Raw: fmt.Sprintf("error: %v", err)}
		}

		action, payload := parseAgentResponse(resp)
		return AgentLLMDoneMsg{Action: action, Payload: payload, Raw: resp}
	}
}

func (m *AppModel) forceAgentAnswerCmd() tea.Cmd {
	question := m.agentQuestion
	accumulated := m.agentCtx.String()
	numSearches := len(m.agentSearches)
	cfg := m.settings.ToLLMConfig()
	cfg.LogDir = m.ragDir

	return func() tea.Msg {
		llmCtx, llmCancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer llmCancel()
		ctxText := accumulated
		if ctxText == "" {
			ctxText = "(no context retrieved)"
		}

		prompt := fmt.Sprintf(
			"You have used all your search steps. Based on the context retrieved, answer this question as best you can. If the context is insufficient, say so.\n\nQuestion: %s\n\nContext:\n%s\n\nAnswer:",
			question, ctxText,
		)
		messages := []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		}
		workers.Logf(m.ragDir, "agent forced answer request searches=%d contextChars=%d", numSearches, len(ctxText))

		resp, err := workers.CallLLM(llmCtx, cfg, messages)
		if err != nil {
			resp = fmt.Sprintf("[LLM error: %v]", err)
		}
		return AgentLLMDoneMsg{
			Action:  "ANSWER",
			Payload: fmt.Sprintf("%s\n\n*(agentic · forced · %d searches)*", resp, numSearches),
		}
	}
}

func (m *AppModel) resetAgentState() {
	m.agentSearches = nil
	m.agentFailed = nil
	m.agentCtx.Reset()
	m.agentStep = 0
	m.agentMaxSteps = -1
	m.agentQuestion = ""
	m.agentConsecFail = 0
	m.streamSources = nil
}

// ─── Streaming ────────────────────────────────────────────────────────────────

// startStreamingCmd wires up the streaming goroutine and returns a cmd to start
// collecting tokens.
func (m *AppModel) startStreamingCmd(retrievedContext string) []tea.Cmd {
	// RAG-only mode: if no API key is configured, skip LLM and show chunks directly.
	if m.settings.NIMAPIKey == "" || m.tuiState.RAGOnly {
		// ── styles ──────────────────────────────────────────────────────────
		accentStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
		dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
		rankStyle := lipgloss.NewStyle().Foreground(ui.ColorGreen).Bold(true)
		scoreStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan)
		starStyle := lipgloss.NewStyle().Foreground(ui.ColorYellow).Bold(true)
		srcStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite).Bold(true)
		textStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite)

		// ── header notice ───────────────────────────────────────────────────
		label := "NO API KEY"
		if m.tuiState.RAGOnly {
			label = "RAG-ONLY MODE"
		}
		n := len(m.streamSources)
		chunkWord := "chunk"
		if n != 1 {
			chunkWord = "chunks"
		}
		notice := accentStyle.Render("◆ "+label) +
			dimStyle.Render(fmt.Sprintf("  %d %s retrieved", n, chunkWord))
		m.addMessage(ChatMessage{Role: "system", Content: notice, Prerendered: true})

		// ── card layout ─────────────────────────────────────────────────────
		cardWidth := m.width - 4
		if cardWidth < 24 {
			cardWidth = 24
		}
		// inner content width = cardWidth - 2 (borders) - 2 (padding each side)
		innerWidth := cardWidth - 6

		var sb strings.Builder
		for i, r := range m.streamSources {
			score := r.RerankScore
			if score == 0 {
				score = r.Score
			}

			// score bar
			const barLen = 10
			filled := clampScoreBarFill(score, barLen)
			bar := scoreStyle.Render(strings.Repeat("█", filled)) +
				dimStyle.Render(strings.Repeat("░", barLen-filled))

			star := dimStyle.Render("·")
			if r.KeywordBoosted {
				star = starStyle.Render("★")
			}

			ts := browserDisplayTimestamp(r.Chunk.TimestampStart)

			// metadata line: #1  ██████░░░░  0.8234  ★  source  timestamp
			meta := rankStyle.Render(fmt.Sprintf("#%-2d", i+1)) + "  " +
				bar + "  " +
				scoreStyle.Render(fmt.Sprintf("%.4f", score)) + "  " +
				star + "  " +
				srcStyle.Render(r.Source) + "  " +
				dimStyle.Render(ts)

			// text: collapse newlines, wrap to innerWidth, max 4 lines
			text := sanitizeBrowserChunkText(r.Chunk.Text, r.Chunk.Sender)
			maxLen := innerWidth * 4
			if maxLen < 80 {
				maxLen = 80
			}
			if len(text) > maxLen {
				text = text[:maxLen] + "…"
			}
			runes := []rune(text)
			var lines []string
			for len(runes) > innerWidth {
				cut := innerWidth
				for cut > 0 && runes[cut] != ' ' {
					cut--
				}
				if cut == 0 {
					cut = innerWidth
				}
				lines = append(lines, string(runes[:cut]))
				runes = []rune(strings.TrimLeft(string(runes[cut:]), " "))
			}
			if len(runes) > 0 {
				lines = append(lines, string(runes))
			}

			sep := dimStyle.Render(strings.Repeat("─", innerWidth))
			body := meta + "\n" + sep + "\n" + textStyle.Render(strings.Join(lines, "\n"))

			borderColor := ui.ColorDim
			if i == 0 {
				borderColor = ui.ColorCyan
			}
			card := lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(borderColor).
				Padding(0, 1).
				Width(cardWidth).
				Render(body)

			if i > 0 {
				sb.WriteString("\n")
			}
			sb.WriteString(card)
		}

		m.appState = StateIdle
		// Store results for the interactive browser.
		m.ragBrowseChunks = make([]workers.Result, len(m.streamSources))
		copy(m.ragBrowseChunks, m.streamSources)
		m.ragBrowseCursor = 0
		m.ragBrowseCtxOpen = false
		m.addMessage(ChatMessage{
			Role:        "assistant",
			Content:     sb.String(),
			Sources:     m.streamSources,
			Prerendered: true,
		})
		m.streamSources = nil
		return nil
	}

	// ── Show chunk cards before LLM answer + enable B-to-browse ────────────
	if len(m.streamSources) > 0 {
		dimStyle2 := lipgloss.NewStyle().Foreground(ui.ColorDim)
		rankStyle2 := lipgloss.NewStyle().Foreground(ui.ColorGreen).Bold(true)
		scoreStyle2 := lipgloss.NewStyle().Foreground(ui.ColorCyan)
		starStyle2 := lipgloss.NewStyle().Foreground(ui.ColorYellow).Bold(true)
		srcStyle2 := lipgloss.NewStyle().Foreground(ui.ColorWhite).Bold(true)
		textStyle2 := lipgloss.NewStyle().Foreground(ui.ColorWhite)

		cardWidth2 := m.width - 4
		if cardWidth2 < 24 {
			cardWidth2 = 24
		}
		innerWidth2 := cardWidth2 - 6

		var chunkSB strings.Builder
		for i, r := range m.streamSources {
			score := r.RerankScore
			if score == 0 {
				score = r.Score
			}
			const barLen = 10
			filled := clampScoreBarFill(score, barLen)
			bar := scoreStyle2.Render(strings.Repeat("█", filled)) +
				dimStyle2.Render(strings.Repeat("░", barLen-filled))
			star := dimStyle2.Render("·")
			if r.KeywordBoosted {
				star = starStyle2.Render("★")
			}
			ts := browserDisplayTimestamp(r.Chunk.TimestampStart)
			meta := rankStyle2.Render(fmt.Sprintf("#%-2d", i+1)) + "  " +
				bar + "  " +
				scoreStyle2.Render(fmt.Sprintf("%.4f", score)) + "  " +
				star + "  " +
				srcStyle2.Render(r.Source) + "  " +
				dimStyle2.Render(ts)
			text := sanitizeBrowserChunkText(r.Chunk.Text, r.Chunk.Sender)
			maxLen2 := innerWidth2 * 3
			if maxLen2 < 80 {
				maxLen2 = 80
			}
			if len(text) > maxLen2 {
				text = text[:maxLen2] + "…"
			}
			runes2 := []rune(text)
			var lines2 []string
			for len(runes2) > innerWidth2 {
				cut := innerWidth2
				for cut > 0 && runes2[cut] != ' ' {
					cut--
				}
				if cut == 0 {
					cut = innerWidth2
				}
				lines2 = append(lines2, string(runes2[:cut]))
				runes2 = []rune(strings.TrimLeft(string(runes2[cut:]), " "))
			}
			if len(runes2) > 0 {
				lines2 = append(lines2, string(runes2))
			}
			sep2 := dimStyle2.Render(strings.Repeat("─", innerWidth2))
			body2 := meta + "\n" + sep2 + "\n" + textStyle2.Render(strings.Join(lines2, "\n"))
			borderColor2 := ui.ColorDim
			if i == 0 {
				borderColor2 = ui.ColorCyan
			}
			card2 := lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(borderColor2).
				Padding(0, 1).
				Width(cardWidth2).
				Render(body2)
			if i > 0 {
				chunkSB.WriteString("\n")
			}
			chunkSB.WriteString(card2)
		}
		hint := dimStyle2.Render(fmt.Sprintf("  %d chunks retrieved  ·  press ", len(m.streamSources))) +
			lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true).Render("B") +
			dimStyle2.Render(" to browse")
		chunkSB.WriteString("\n" + hint)
		m.addMessage(ChatMessage{Role: "system", Content: chunkSB.String(), Prerendered: true})
		// Populate browser so B works immediately.
		m.ragBrowseChunks = make([]workers.Result, len(m.streamSources))
		copy(m.ragBrowseChunks, m.streamSources)
		m.ragBrowseCursor = 0
		m.ragBrowseCtxOpen = false
	}

	// Find the query from the last user message.
	query := ""
	for i := len(m.messages) - 1; i >= 0; i-- {
		if m.messages[i].Role == "user" {
			query = m.messages[i].Content
			break
		}
	}

	prompt := buildPrompt(query, retrievedContext, "", m.tuiState.OutputMode)
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: prompt},
	}

	cfg := m.settings.ToLLMConfig()
	cfg.LogDir = m.ragDir
	workers.Logf(
		m.ragDir,
		"streaming llm request query=%q contextChars=%d mode=%s model=%s",
		query,
		len(retrievedContext),
		m.tuiState.OutputMode,
		m.settings.NIMModel,
	)
	// Create fresh channels for this streaming session to avoid
	// "close of closed channel" panics when a previous session's channel
	// was already closed by StreamLLM on error or EOF.
	tokenCh := make(chan string, 100)
	errCh := make(chan error, 1)
	m.tokenChan = tokenCh
	m.errChan = errCh
	m.appState = StateStreaming
	// Ensure we have a valid cancellable context for this stream.
	if m.activeCtx == nil || m.activeCtx.Err() != nil {
		m.activeCtx, m.cancelFn = context.WithCancel(context.Background())
	}

	return []tea.Cmd{
		func() tea.Msg {
			go workers.StreamLLM(m.activeCtx, cfg, messages, tokenCh, errCh)
			return nil
		},
		waitForTokenCmd(tokenCh, errCh),
	}
}

// ─── Rendering ────────────────────────────────────────────────────────────────

func (m *AppModel) addMessage(msg ChatMessage) {
	m.messages = append(m.messages, msg)
	m.renderMessages()
	m.viewport.GotoBottom()
}

func (m *AppModel) renderStreamingMessage() {
	content := m.streamBuf.String()
	// When the LLM is emitting rich JSON, show a placeholder instead of raw JSON.
	if IsStreamingJSON(content) {
		placeholder := lipgloss.NewStyle().Foreground(ui.ColorDim).Italic(true).Render("⟳  rendering structured output…")
		if len(m.messages) > 0 && m.messages[len(m.messages)-1].Role == "assistant" {
			m.messages[len(m.messages)-1].Content = placeholder
			m.messages[len(m.messages)-1].Streaming = true
		} else {
			m.messages = append(m.messages, ChatMessage{
				Role:      "assistant",
				Content:   placeholder,
				Streaming: true,
			})
		}
	} else if len(m.messages) > 0 && m.messages[len(m.messages)-1].Role == "assistant" {
		m.messages[len(m.messages)-1].Content = content
		m.messages[len(m.messages)-1].Streaming = true
	} else {
		m.messages = append(m.messages, ChatMessage{
			Role:      "assistant",
			Content:   content,
			Streaming: true,
		})
	}
	m.renderMessages()
	m.viewport.GotoBottom()
}

func (m *AppModel) renderMessages() {
	var sb strings.Builder
	for _, msg := range m.messages {
		sb.WriteString(m.renderMessage(msg))
		sb.WriteString("\n")
	}
	m.viewport.SetContent(sb.String())
}

func (m AppModel) renderMessage(msg ChatMessage) string {
	switch msg.Role {
	case "user":
		header := ui.UserMsgStyle.Render("[You]")
		return header + " " + msg.Content

	case "assistant":
		var sb strings.Builder
		label := "[AI]"
		if len(msg.Sources) > 0 || len(msg.Searches) > 0 {
			label = "[RAG]"
		}
		var header string
		if msg.Streaming {
			header = ui.AIStreamingMsgStyle.Render(label)
		} else {
			header = ui.AssistantMsgStyle.Render(label)
		}
		rendered := msg.Content
		// Skip markdown rendering for pre-rendered lipgloss content.
		if !msg.Prerendered && m.mdRenderer != nil && !msg.Streaming {
			if md, err := m.mdRenderer.Render(msg.Content); err == nil {
				rendered = strings.TrimRight(md, "\n")
			}
		}
		if msg.Prerendered {
			// Cards already include full styling — just show them without header/wrap.
			sb.WriteString("\n" + rendered)
		} else {
			sb.WriteString(header + " " + rendered)
		}
		if m.tuiState.ShowSources && len(msg.Sources) > 0 {
			sb.WriteString("\n")
			sb.WriteString(renderSourceTable(msg.Sources, m.tuiState.Window))
		}
		return sb.String()

	case "agentic_step":
		return ui.AgentStepStyle.Render("  " + msg.Content)

	case "system":
		if msg.Prerendered {
			return "  " + msg.Content
		}
		return ui.SystemMsgStyle.Render("↦ " + msg.Content)

	case "error":
		return ui.ErrorMsgStyle.Render("[ERROR] " + msg.Content)
	}
	return msg.Content
}

func renderSourceTable(results []workers.Result, window int) string {
	if len(results) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString(ui.SourceTableStyle.Render("Sources:") + "\n")
	header := fmt.Sprintf("  %-4s %-7s %-3s %-7s %-20s %s", "#", "Score", "★", "Src", "Time", "Text")
	sb.WriteString(ui.SystemMsgStyle.Render(header) + "\n")
	for _, r := range results {
		score := r.RerankScore
		if score == 0 {
			score = r.Score
		}
		star := " "
		if r.KeywordBoosted {
			star = "★"
		}
		neighbor := " "
		if r.IsNeighbor {
			neighbor = "N"
		}
		ts := r.Chunk.TimestampStart
		if len(ts) > 19 {
			ts = ts[:19]
		}
		text := r.Chunk.Text
		if len(text) > 55 {
			text = text[:55] + "…"
		}
		src := r.Source
		if len(src) > 6 {
			src = src[:6]
		}
		row := fmt.Sprintf("  %-4d %-7.3f %-3s %-7s %-20s %s", r.Rank, score, star+neighbor, src, ts, text)
		sb.WriteString(ui.SystemMsgStyle.Render(row) + "\n")

		// Show context window (surrounding messages) when available and window > 0.
		if window > 0 && len(r.ContextWindow) > 0 {
			sb.WriteString(ui.HelpStyle.Render("     ╭─ context window") + "\n")
			anchorID := r.Chunk.ChunkID
			for _, wc := range r.ContextWindow {
				wts := wc.TimestampStart
				if len(wts) > 16 {
					wts = wts[:16]
				}
				wtext := wc.Text
				if len(wtext) > 60 {
					wtext = wtext[:60] + "…"
				}
				sender := wc.Sender
				if len(sender) > 10 {
					sender = sender[:10]
				}
				line := fmt.Sprintf("     │ [%s] %s: %s", wts, sender, wtext)
				if wc.ChunkID == anchorID {
					sb.WriteString(ui.ContextAnchorStyle.Render("     ▶ ["+wts+"] "+sender+": "+wtext) + "\n")
				} else {
					sb.WriteString(ui.HelpStyle.Render(line) + "\n")
				}
			}
			sb.WriteString(ui.HelpStyle.Render("     ╰─") + "\n")
		}
	}
	return sb.String()
}

// ─── View ─────────────────────────────────────────────────────────────────────

func (m AppModel) View() string {
	if m.width == 0 {
		return "Loading…"
	}
	return m.viewMain()
}

// viewMain is the single top-level layout: content area + input row + status bar.
// The status bar is always at the bottom; the content area changes by screen.
func (m AppModel) viewMain() string {
	var content string
	switch m.screen {
	case ScreenSettings:
		content = m.viewSettingsContent()
	case ScreenSettingsMenu:
		content = m.viewSettingsMenuContent()
	case ScreenRetrievalSettings:
		content = m.viewRetrievalSettingsContent()
	case ScreenAPISettings:
		content = m.viewAPISettingsContent()
	case ScreenInterfaceSettings:
		content = m.viewInterfaceSettingsContent()
	case ScreenPipeline:
		content = m.viewPipelineContent()
	case ScreenChatList:
		content = m.viewChatListContent()
	case ScreenChatViewer:
		content = m.viewChatViewerContent()
	case ScreenRagBrowser:
		content = m.viewRagBrowserContent()
	case ScreenChats:
		content = m.viewChatsContent()
	case ScreenHelp:
		content = m.viewHelpContent()
	case ScreenModelPicker:
		content = m.viewModelPickerContent()
	case ScreenNick:
		content = m.viewNickContent()
	case ScreenIngestLog:
		content = m.viewIngestLogContent()
	default: // ScreenChat
		if m.appState == StateStarting || len(m.messages) == 0 {
			content = m.viewSplash()
		} else {
			content = m.viewport.View()
		}
	}
	inputRow := m.renderInputRow()
	statusBar := m.renderStatusBar()
	ingestBar := m.renderIngestBar()
	if m.clarify.Active {
		clarifyPanel := RenderClarify(&m.clarify, m.width)
		if ingestBar != "" {
			return lipgloss.JoinVertical(lipgloss.Left, content, clarifyPanel, inputRow, ingestBar, statusBar)
		}
		return lipgloss.JoinVertical(lipgloss.Left, content, clarifyPanel, inputRow, statusBar)
	}
	if ingestBar != "" {
		return lipgloss.JoinVertical(lipgloss.Left, content, inputRow, ingestBar, statusBar)
	}
	return lipgloss.JoinVertical(lipgloss.Left, content, inputRow, statusBar)
}

// renderIngestBar returns a two-line progress widget shown during ingest.
// Top row: mauve — spinner + filename + current step message.
// Bottom row: green — ASCII progress bar + percentage.
// Returns "" when no ingest is in progress or when the log viewer is open
// (the log viewer is the authoritative ingest UI on that screen).
func (m AppModel) renderIngestBar() string {
	if !m.ingestInProgress || m.screen == ScreenIngestLog {
		return ""
	}
	mauveBg := lipgloss.Color("#4a383f")
	mauveText := lipgloss.Color("#e0b8c8")
	greenBg := lipgloss.Color("#2e3e30")
	greenFill := ui.ColorCyan // sage green #b8d8ba
	greenDim := ui.ColorDim

	stem := strings.TrimSuffix(m.ingestFilename, filepath.Ext(m.ingestFilename))
	msg := m.ingestMessage
	if msg == "" {
		msg = "preparing…"
	}

	// Row 1: spinner + label — build as plain then pad explicitly so we don't
	// nest pre-rendered ANSI inside Width(), which causes bg transparency gaps.
	labelPlain := fmt.Sprintf("  %s  ingesting %s  —  %s", "◌", stem, msg)
	labelRendered := lipgloss.NewStyle().Foreground(mauveText).Background(mauveBg).
		Render(fmt.Sprintf("  %s  ingesting %s  —  %s", m.spinner.View(), stem, msg))
	pad1 := m.width - len([]rune(labelPlain)) // rune count ≈ display width for these chars
	if pad1 < 0 {
		pad1 = 0
	}
	row1 := labelRendered + lipgloss.NewStyle().Background(mauveBg).Render(strings.Repeat(" ", pad1))

	// Row 2: progress bar — each segment rendered separately then padded.
	pct := m.ingestProgress
	barW := m.width - 10 // leave room for "  NNN%  " suffix (8 chars)
	if barW < 4 {
		barW = 4
	}
	filled := barW * pct / 100
	if filled > barW {
		filled = barW
	}
	filledStr := lipgloss.NewStyle().Foreground(greenFill).Background(greenBg).Render(strings.Repeat("█", filled))
	emptyStr := lipgloss.NewStyle().Foreground(greenDim).Background(greenBg).Render(strings.Repeat("░", barW-filled))
	pctStr := lipgloss.NewStyle().Foreground(greenFill).Background(greenBg).Bold(true).Render(fmt.Sprintf("  %3d%%  ", pct))
	// Pad remainder so background fills the full row.
	usedW := 2 + barW + 8 // "  " + bar + "  NNN%  "
	padW := m.width - usedW
	if padW < 0 {
		padW = 0
	}
	prefix2 := lipgloss.NewStyle().Background(greenBg).Render("  ")
	trail2 := lipgloss.NewStyle().Background(greenBg).Render(strings.Repeat(" ", padW))
	row2 := prefix2 + filledStr + emptyStr + pctStr + trail2

	return lipgloss.JoinVertical(lipgloss.Left, row1, row2)
}

// refreshIngestLog rebuilds the ingest log viewport content (log lines only; title is pinned outside).
func (m *AppModel) refreshIngestLog() {
	dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
	var sb strings.Builder
	for _, line := range m.ingestLog {
		sb.WriteString(dimStyle.Render(line) + "\n")
	}
	if m.ingestInProgress {
		sb.WriteString("\n" + dimStyle.Render("Still running… ESC to return"))
	} else {
		sb.WriteString("\n" + dimStyle.Render("Ingest complete. ESC to return"))
	}
	m.ingestLogVP.SetContent(sb.String())
	m.ingestLogVP.GotoBottom()
}

func (m AppModel) ingestLogVpHeight() int {
	// ScreenIngestLog layout (no ingest bar on this screen):
	//   border(2) + title(1) + sep(1) + viewport(h) + sep(1) + footer(1) + inputBox(3) + statusBar(1) = m.height
	// So viewport(h) = m.height - 10
	h := m.height - 10
	if h < 3 {
		return 3
	}
	return h
}

func (m AppModel) viewIngestLogContent() string {
	titleStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
	dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
	accentStyle := lipgloss.NewStyle().Foreground(ui.ColorGreen).Bold(true)
	warnStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#ef959d"))
	label := m.ingestFilename
	if label == "" {
		label = "pending"
	}

	// Pinned header: spinner (when running) + title + file
	var titleLine string
	if m.ingestInProgress {
		pctStr := accentStyle.Render(fmt.Sprintf("%d%%", m.ingestProgress))
		titleLine = m.spinner.View() + "  " + titleStyle.Render("Live Ingest Log") +
			"  " + dimStyle.Render(label) + "  " + pctStr
	} else {
		titleLine = accentStyle.Render("✓") + "  " + titleStyle.Render("Ingest Complete") +
			"  " + dimStyle.Render(label)
	}
	sep := dimStyle.Render(strings.Repeat("─", m.width-10))

	// Footer hint
	var footerLine string
	if m.ingestInProgress {
		footerLine = dimStyle.Render("  ↑↓ scroll · ESC back · ") +
			warnStyle.Render("[C]") + dimStyle.Render(" cancel")
	} else {
		footerLine = dimStyle.Render("  ↑↓ scroll · ESC to return")
	}

	inner := titleLine + "\n" + sep + "\n" + m.ingestLogVP.View() + "\n" + sep + "\n" + footerLine
	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(ui.ColorCyan).
		Padding(0, 1).
		Width(m.width - 4).
		Render(inner)
}

func (m *AppModel) handleIngestLogKey(msg tea.KeyMsg) []tea.Cmd {
	m.acConsumedKey = true
	switch msg.Type {
	case tea.KeyEsc:
		m.screen = ScreenChat
		return nil
	case tea.KeyRunes:
		switch string(msg.Runes) {
		case "c", "C":
			if m.ingestInProgress {
				// Kill the bridge process to abort the running ingest.
				if m.bridge != nil {
					m.bridge.Kill()
					m.bridge = nil
					m.bridgeReady = false
				}
				m.ingestInProgress = false
				m.ingestProgress = 0
				m.viewport.Height = m.viewportHeight()
				m.ingestLogVP.Height = m.ingestLogVpHeight()
				m.ingestLog = append(m.ingestLog, "[CANCELLED] Ingest stopped by user.")
				m.refreshIngestLog()
				m.screen = ScreenChat
				m.addMessage(ChatMessage{Role: "system", Content: "⚠ Ingest cancelled — restarting bridge…"})
				return []tea.Cmd{restartBridgeCmd(m.ragDir)}
			}
		}
	}
	var cmd tea.Cmd
	m.ingestLogVP, cmd = m.ingestLogVP.Update(msg)
	return []tea.Cmd{cmd}
}

func (m AppModel) renderStatusBar() string {
	chatName := m.activeChat
	if chatName == "" {
		chatName = "none"
	}

	barBg := ui.ColorDim
	// barText styles plain key=value items — explicit bg ensures no transparent
	// gaps after inner ANSI resets from other pill styles.
	barText := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#c8a8b0")).
		Background(barBg)
	// barState styles busy-indicator text (cyan fg, same bg).
	barState := lipgloss.NewStyle().
		Foreground(ui.ColorCyan).
		Background(barBg).
		Bold(true)
	// barErr styles error indicators (red fg, same bg).
	barErr := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#f28b82")).
		Background(barBg).
		Bold(true)
	// barSep is the two-space gap between items — must carry the bar bg.
	barSep := barText.Render("  ")

	activeFlag := lipgloss.NewStyle().
		Foreground(ui.ColorDim).
		Background(ui.ColorGreen).
		Bold(true).
		Padding(0, 1)
	modelShort := m.settings.NIMModel
	if idx := strings.LastIndex(modelShort, "/"); idx >= 0 {
		modelShort = modelShort[idx+1:]
	}

	// Build parts — always-visible items first (all with explicit bg).
	parts := []string{
		barText.Render("chat=" + chatName),
		barText.Render("model=" + modelShort),
		barText.Render(fmt.Sprintf("window=%d", m.tuiState.Window)),
		barText.Render(fmt.Sprintf("k=%d", m.settings.FinalK)),
	}

	// Flags shown only when active.
	if m.tuiState.Debug {
		parts = append(parts, activeFlag.Render("debug"))
	}
	if m.tuiState.ShowSources {
		parts = append(parts, activeFlag.Render("sources"))
	}
	if m.tuiState.AgentMode {
		parts = append(parts, activeFlag.Render("agent"))
	}
	if m.tuiState.Confident {
		parts = append(parts, activeFlag.Render("confident"))
	}
	if m.settings.ThinkingMode {
		parts = append(parts, activeFlag.Render("thinking"))
	}
	if m.tuiState.OutputMode != "" && m.tuiState.OutputMode != "plain" {
		parts = append(parts, activeFlag.Render("mode="+m.tuiState.OutputMode))
	}

	// Busy state indicator — same bg so no transparent gap.
	switch m.appState {
	case StateStarting:
		parts = append(parts, barState.Render("[starting…]"))
	case StateRetrieving:
		parts = append(parts, barState.Render("[retrieving…]"))
	case StateStreaming:
		parts = append(parts, barState.Render("[streaming…]"))
	case StateAgentStep:
		parts = append(parts, barState.Render(fmt.Sprintf("[agent step %d]", m.agentStep+1)))
	case StateError:
		parts = append(parts, barErr.Render("[error]"))
	}
	if m.needsAPIKeyWarning() {
		parts = append(parts, barErr.Render("[set NIM API key or /rag]"))
	}

	bar := strings.Join(parts, barSep)

	// Update badge — right-aligned, shown when a newer version is available.
	if m.updateAvailable != "" {
		badge := lipgloss.NewStyle().
			Foreground(ui.ColorYellow).Bold(true).Background(barBg).
			Render(fmt.Sprintf("↑ %s available · /update", m.updateAvailable))
		barPlain := lipgloss.NewStyle().Width(m.width).Render(bar)
		barWidth := lipgloss.Width(barPlain)
		badgeWidth := lipgloss.Width(badge)
		if barWidth+badgeWidth+2 <= m.width {
			bar = bar + barText.Render(strings.Repeat(" ", m.width-barWidth-badgeWidth)) + badge
		} else {
			bar = bar + barSep + badge
		}
	}

	// Version badge — far right of status bar, debug mode on main chat screen only.
	if m.tuiState.Debug && m.screen == ScreenChat {
		verBadge := lipgloss.NewStyle().Foreground(ui.ColorDim).Faint(true).Background(barBg).Render(AppVersion)
		barNaturalWidth := lipgloss.Width(bar)
		verW := lipgloss.Width(verBadge)
		padding := m.width - barNaturalWidth - verW - 2 // -2 for status bar padding
		if padding > 0 {
			bar = bar + barText.Render(strings.Repeat(" ", padding)) + verBadge
		}
	}

	return ui.StatusBarStyle.Width(m.width).Render(bar)
}

func (m AppModel) needsAPIKeyWarning() bool {
	return strings.TrimSpace(m.settings.NIMAPIKey) == "" && !m.tuiState.RAGOnly
}

func (m *AppModel) maybeWarnMissingAPIKey() {
	if m.startupAPIWarned || !m.needsAPIKeyWarning() {
		return
	}
	msg := lipgloss.NewStyle().Foreground(ui.ColorRed).Bold(true).Render("NIM API key not set.") +
		lipgloss.NewStyle().Foreground(ui.ColorDim).Render(" AI answers stay off until you add one in Settings > API or toggle /rag.")
	m.addMessage(ChatMessage{Role: "system", Content: msg, Prerendered: true})
	m.startupAPIWarned = true
}

func (m AppModel) renderInputRow() string {
	var prefix string
	switch {
	case m.appState != StateIdle && m.appState != StateError:
		prefix = m.spinner.View() + " "
	default:
		prefix = ui.UserMsgStyle.Render("> ")
	}

	// Border colour: amber when AI is active, yellow when autocomplete, cyan normally.
	borderColor := ui.ColorCyan
	switch m.appState {
	case StateRetrieving, StateStreaming, StateAgentStep:
		borderColor = ui.ColorAIActive
	}
	if m.acActive {
		borderColor = ui.ColorYellow
	}

	inputContent := prefix + m.textarea.View()
	inputBox := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(borderColor).
		Width(m.width - 4).
		Render(inputContent)

	var parts []string
	if m.screen == ScreenHelp {
		hint := "  ↑↓ choose category · Enter open · Esc close"
		if m.helpSection != "" {
			hint = "  ↑↓ PgUp/PgDn scroll · Esc/back return"
		}
		parts = append(parts, ui.HelpStyle.Render(hint))
	}
	parts = append(parts, inputBox)
	if m.acActive && len(m.acSuggestions) > 0 {
		parts = append(parts, m.renderAutocomplete())
	}
	return lipgloss.JoinVertical(lipgloss.Left, parts...)
}

func (m AppModel) viewSettingsContent() string {
	var sb strings.Builder
	sb.WriteString(ui.TitleStyle.Render("Model Settings") + "\n\n")

	for i, row := range settingsRows {
		isSelected := i == m.settingsCursor
		marker := "  "
		numStr := fmt.Sprintf("[%d]", i+1)
		if isSelected {
			marker = ui.ActiveFlagStyle.Render("▶ ")
			numStr = ui.TitleStyle.Render(numStr)
		} else {
			numStr = ui.SystemMsgStyle.Render(numStr)
		}

		var valStr string
		if row.key == m.settingsEditing {
			// Show the value currently being typed with a cursor.
			input := m.settingsInput
			if row.key == "api_key" && input == "" {
				input = "(enter new key)"
			}
			valStr = ui.ActiveFlagStyle.Render(input + "_")
		} else {
			val := m.settingsValue(row.key)
			if row.key == "thinking" || row.key == "rag_only" {
				var on bool
				if row.key == "thinking" {
					on = m.settings.ThinkingMode
				} else {
					on = m.tuiState.RAGOnly
				}
				if on {
					valStr = ui.ActiveFlagStyle.Render("ON")
				} else {
					valStr = ui.SystemMsgStyle.Render("off")
				}
			} else if isSelected {
				valStr = lipgloss.NewStyle().Foreground(ui.ColorWhite).Render(val)
			} else {
				valStr = ui.SystemMsgStyle.Render(val)
			}
		}
		sb.WriteString(fmt.Sprintf("%s%s %-15s  %s\n", marker, numStr, row.label, valStr))
	}
	sb.WriteString("\n")
	if m.settingsEditing != "" {
		sb.WriteString(ui.HelpStyle.Render("Enter to save · ESC to cancel"))
	} else {
		sb.WriteString(ui.HelpStyle.Render("↑↓: navigate · Enter: edit/toggle · ESC: back"))
	}

	return lipgloss.NewStyle().Height(m.viewportHeight()).Render(
		ui.BorderStyle.Width(m.width - 4).Render(sb.String()),
	)
}

// settingsValue returns the current display string for a settings field key.
func (m AppModel) settingsValue(key string) string {
	switch key {
	case "model":
		return m.settings.NIMModel
	case "temperature":
		return fmt.Sprintf("%g", m.settings.Temperature)
	case "top_p":
		return fmt.Sprintf("%g", m.settings.TopP)
	case "max_tokens":
		return strconv.Itoa(m.settings.MaxTokens)
	case "api_key":
		return maskKey(m.settings.NIMAPIKey)
	case "base_url":
		return m.settings.NIMBaseURL
	case "thinking":
		if m.settings.ThinkingMode {
			return "ON"
		}
		return "off"
	case "rag_only":
		if m.tuiState.RAGOnly {
			return "ON"
		}
		return "off"
	case "nick":
		userCount := len(m.modelNicknames)
		if userCount > 0 {
			return fmt.Sprintf("%d built-in, %d custom", len(defaultNicknames), userCount)
		}
		return fmt.Sprintf("%d built-in", len(defaultNicknames))
	}
	return ""
}

func (m AppModel) viewChatsContent() string {
	var sb strings.Builder
	sb.WriteString(ui.TitleStyle.Render("Chats") + "\n\n")

	if len(m.chatSlugList) == 0 {
		sb.WriteString(ui.SystemMsgStyle.Render("No chats found.") + "\n")
	} else {
		for i, slug := range m.chatSlugList {
			isActive := slug == m.activeChat
			isCursor := i == m.chatCursor
			marker := "  "
			if isCursor {
				marker = ui.ActiveFlagStyle.Render("▶ ")
			}
			displayName := slug
			if info, ok := m.chatList[slug]; ok {
				if infoMap, ok := info.(map[string]interface{}); ok {
					if dn, ok := infoMap["display_name"].(string); ok {
						displayName = dn
					}
				}
			}
			suffix := ""
			if isActive {
				suffix = " " + ui.ActiveFlagStyle.Render("(active)")
			}
			sb.WriteString(fmt.Sprintf("%s[%d] %s%s\n", marker, i+1, displayName, suffix))
		}
	}
	sb.WriteString("\n")
	sb.WriteString(ui.HelpStyle.Render("↑↓: navigate · Enter: switch · number key: quick switch · ESC: back"))

	return lipgloss.NewStyle().Height(m.viewportHeight()).Render(
		ui.BorderStyle.Width(m.width - 4).Render(sb.String()),
	)
}

func (m AppModel) viewHelpContent() string {
	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(ui.ColorCyan).
		Padding(0, 1).
		Width(m.width - 4).
		Render(m.helpVp.View())
}

func (m *AppModel) refreshHelpContent() {
	if m.helpSection == "" {
		m.helpVp.SetContent(m.buildHelpContent())
		return
	}
	m.helpVp.SetContent(m.buildHelpSectionContent(m.helpSection))
}

// buildHelpContent returns the interactive help category menu.
func (m AppModel) buildHelpContent() string {
	titleStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
	sectionStyle := lipgloss.NewStyle().Foreground(ui.ColorGreen).Bold(true)
	dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
	selectedStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite).Bold(true)
	cursorStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true).Width(3)
	plainCursor := lipgloss.NewStyle().Foreground(ui.ColorDim).Width(3)

	var sb strings.Builder
	sb.WriteString(titleStyle.Render("ragtag · Help") + "\n\n")
	sb.WriteString(dimStyle.Render("Choose a category and press Enter.\n\n"))

	for i, section := range helpSections {
		if i == m.helpCursor {
			sb.WriteString(cursorStyle.Render("▶") + selectedStyle.Render(section.title) + "\n")
			sb.WriteString(plainCursor.Render("") + lipgloss.NewStyle().Foreground(ui.ColorWhite).Render(section.summary) + "\n\n")
		} else {
			sb.WriteString(plainCursor.Render("") + sectionStyle.Render(section.title) + "\n")
			sb.WriteString(plainCursor.Render("") + dimStyle.Render(section.summary) + "\n\n")
		}
	}
	sb.WriteString(dimStyle.Render("Tip: /help opens this menu anytime. /clear wipes the visible chat log only."))
	return sb.String()
}

func (m AppModel) buildHelpSectionContent(sectionID string) string {
	titleStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
	sectionStyle := lipgloss.NewStyle().Foreground(ui.ColorGreen).Bold(true)
	cmdStyle := lipgloss.NewStyle().Foreground(ui.ColorYellow)
	dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)

	var sb strings.Builder
	switch sectionID {
	case "getting-started":
		sb.WriteString(titleStyle.Render("Help · Getting Started") + "\n\n")
		sb.WriteString("1. Ask normally in the main chat box. Ragtag retrieves matching chunks, then asks the model to answer from them.\n")
		sb.WriteString("2. Toggle " + cmdStyle.Render("/agent") + " when you want multi-step searching instead of one retrieval pass.\n")
		sb.WriteString("3. Use " + cmdStyle.Render("/rag") + " for retrieval-only mode when you want to inspect chunks without calling the LLM.\n")
		sb.WriteString("4. Use " + cmdStyle.Render("/sources") + " and " + cmdStyle.Render("/debug") + " when you want to see what the system actually retrieved.\n\n")
		sb.WriteString(sectionStyle.Render("Good first commands") + "\n")
		sb.WriteString("  " + cmdStyle.Render("/help") + "          open this help browser\n")
		sb.WriteString("  " + cmdStyle.Render("/model") + "         pick a model interactively\n")
		sb.WriteString("  " + cmdStyle.Render("/window 20") + "     show more surrounding context than the default\n")
		sb.WriteString("  " + cmdStyle.Render("/clear") + "         clear the visible conversation\n")
	case "commands":
		sb.WriteString(titleStyle.Render("Help · Commands") + "\n\n")
		sb.WriteString(sectionStyle.Render("Core flow") + "\n")
		sb.WriteString("  " + cmdStyle.Render("/agent") + "            toggle agent mode\n")
		sb.WriteString("  " + cmdStyle.Render("/agent N query") + "    run one agentic query with a step cap\n")
		sb.WriteString("  " + cmdStyle.Render("/pause") + "            stop the current retrieval or stream\n")
		sb.WriteString("  " + cmdStyle.Render("/clear") + "            clear the chat log in the TUI\n\n")
		sb.WriteString(sectionStyle.Render("Output and visibility") + "\n")
		sb.WriteString("  " + cmdStyle.Render("/mode") + "             cycle plain / structured / rich\n")
		sb.WriteString("  " + cmdStyle.Render("/sources") + "          show source tables under answers\n")
		sb.WriteString("  " + cmdStyle.Render("/debug") + "            show diagnostics and log path\n")
		sb.WriteString("  " + cmdStyle.Render("/tools") + "            inspect agent tool toggles\n\n")
		sb.WriteString(sectionStyle.Render("Retrieval and data") + "\n")
		sb.WriteString("  " + cmdStyle.Render("/window N") + "         set context-window size\n")
		sb.WriteString("  " + cmdStyle.Render("/k N") + "              set chunks sent to the model\n")
		sb.WriteString("  " + cmdStyle.Render("/minresults N") + "     adaptive floor for kept results\n")
		sb.WriteString("  " + cmdStyle.Render("/threshold F") + "     explicit rerank-score cutoff\n")
		sb.WriteString("  " + cmdStyle.Render("/pipeline") + "         open ingestion/rebuild tools\n")
		sb.WriteString("  " + cmdStyle.Render("/ingest") + "           import a file from raw/\n")
		sb.WriteString("  " + cmdStyle.Render("/view") + "             browse raw exported chat files\n")
		sb.WriteString("  " + cmdStyle.Render("/chats") + "            switch active indexed chat\n")
	case "features":
		sb.WriteString(titleStyle.Render("Help · Features & Modes") + "\n\n")
		sb.WriteString(sectionStyle.Render("Agent mode") + "\n")
		sb.WriteString("Agent mode loops through SEARCH → retrieve → decide again. It is best when the answer needs a few different search angles.\n\n")
		sb.WriteString(sectionStyle.Render("Chunk browser") + "\n")
		sb.WriteString("In RAG-only mode, press " + cmdStyle.Render("B") + " after results appear to open the chunk browser. Use ↑↓ to move and Enter to inspect the matched chunk with surrounding context.\n\n")
		sb.WriteString(sectionStyle.Render("Output modes") + "\n")
		sb.WriteString("Plain keeps answers simple. Structured asks for summary + key points JSON. Rich asks for components like chat logs, lists, timelines, and tables.\n\n")
		sb.WriteString(sectionStyle.Render("Update flow") + "\n")
		sb.WriteString("When a newer version exists, ragtag shows a startup prompt and a status-bar notice. " + cmdStyle.Render("/update") + " downloads and installs it.\n")
	case "tips":
		sb.WriteString(titleStyle.Render("Help · Tips & Tricks") + "\n\n")
		sb.WriteString("• Ask short, concrete questions first. Semantic retrieval usually works better with direct phrasing than with long prompts.\n")
		sb.WriteString("• If answers feel shallow, raise " + cmdStyle.Render("/window") + " before raising " + cmdStyle.Render("/k") + ". More nearby chat often helps more than more unrelated chunks.\n")
		sb.WriteString("• Use " + cmdStyle.Render("/debug") + " when something feels wrong. It now points you to ragtag.log for deeper diagnostics.\n")
		sb.WriteString("• If agent mode feels too eager or too broad, inspect " + cmdStyle.Render("/tools") + " and disable context expansion or search temporarily.\n")
		sb.WriteString("• " + cmdStyle.Render("/clear") + " only clears the visible TUI conversation. It does not delete indexed chats or raw files.\n")
	case "models":
		sb.WriteString(titleStyle.Render("Help · Models & Retrieval") + "\n\n")
		sb.WriteString("Default model: " + cmdStyle.Render("openai/gpt-oss-120b") + "\n")
		sb.WriteString("Default window: " + cmdStyle.Render("10") + "\n\n")
		sb.WriteString(sectionStyle.Render("Model picking") + "\n")
		for i, mdl := range allModels {
			sb.WriteString(fmt.Sprintf("  %s %s\n", cmdStyle.Render(fmt.Sprintf("[%2d]", i+1)), mdl))
		}
		sb.WriteString("\n" + sectionStyle.Render("How retrieval knobs interact") + "\n")
		sb.WriteString("  " + cmdStyle.Render("/window") + " changes how much surrounding chat is attached to each hit.\n")
		sb.WriteString("  " + cmdStyle.Render("/k") + " changes how many final chunks survive reranking.\n")
		sb.WriteString("  " + cmdStyle.Render("/minresults") + " keeps a minimum number of results even when filtering is harsh.\n")
		sb.WriteString("  " + cmdStyle.Render("/threshold") + " discards low-score results when you want cleaner context.\n")
	}
	sb.WriteString("\n" + dimStyle.Render("Esc: back to categories"))
	return sb.String()
}

func (m *AppModel) handleNickKey(msg tea.KeyMsg) []tea.Cmd {
	m.acConsumedKey = true

	if m.nickStep == 1 {
		// Step 1: typing the alias name.
		switch msg.Type {
		case tea.KeyEsc:
			m.nickStep = 0
			m.nickInput = ""
		case tea.KeyEnter:
			nick := strings.ToLower(strings.TrimSpace(m.nickInput))
			mdl := allModels[m.nickModelIdx]
			if nick == "" {
				// Empty = clear any user nickname pointing to this model.
				for k, v := range m.modelNicknames {
					if v == mdl {
						delete(m.modelNicknames, k)
					}
				}
			} else {
				m.modelNicknames[nick] = mdl
			}
			_ = m.saveNicknames()
			m.nickStep = 0
			m.nickInput = ""
		case tea.KeyBackspace, tea.KeyDelete:
			if len(m.nickInput) > 0 {
				runes := []rune(m.nickInput)
				m.nickInput = string(runes[:len(runes)-1])
			}
		default:
			if msg.Type == tea.KeyRunes {
				m.nickInput += msg.String()
			}
		}
		return nil
	}

	// Step 0: pick which model to nickname.
	switch msg.Type {
	case tea.KeyEsc:
		m.screen = ScreenSettings
		m.nickStep = 0
	case tea.KeyUp:
		if m.nickModelIdx > 0 {
			m.nickModelIdx--
		}
	case tea.KeyDown:
		if m.nickModelIdx < len(allModels)-1 {
			m.nickModelIdx++
		}
	case tea.KeyEnter:
		// Pre-fill any existing user nickname for this model.
		mdl := allModels[m.nickModelIdx]
		m.nickInput = ""
		for k, v := range m.modelNicknames {
			if v == mdl {
				m.nickInput = k
				break
			}
		}
		m.nickStep = 1
	}
	return nil
}

func (m AppModel) viewNickContent() string {
	var sb strings.Builder
	sb.WriteString(ui.TitleStyle.Render("Nickname a Model") + "\n\n")

	if m.nickStep == 0 {
		for i, mdl := range allModels {
			isSelected := i == m.nickModelIdx
			marker := "  "
			if isSelected {
				marker = ui.ActiveFlagStyle.Render("▶ ")
			}
			numStr := ui.SystemMsgStyle.Render(fmt.Sprintf("[%2d]", i+1))

			nicks := m.nicknamesFor(mdl)
			nickStr := ""
			if len(nicks) > 0 {
				nickStr = "  " + ui.HelpStyle.Render("("+strings.Join(nicks, ", ")+")")
			}

			var modelStr string
			if isSelected {
				modelStr = lipgloss.NewStyle().Foreground(ui.ColorWhite).Bold(true).Render(mdl)
			} else {
				modelStr = lipgloss.NewStyle().Foreground(ui.ColorWhite).Render(mdl)
			}
			sb.WriteString(fmt.Sprintf("%s%s %s%s\n", marker, numStr, modelStr, nickStr))
		}
		sb.WriteString("\n")
		sb.WriteString(ui.HelpStyle.Render("↑↓: navigate · Enter: set nickname · ESC: back to settings"))
	} else {
		mdl := allModels[m.nickModelIdx]
		sb.WriteString(fmt.Sprintf("Model: %s\n\n", ui.TitleStyle.Render(mdl)))

		var builtins []string
		for k, v := range defaultNicknames {
			if v == mdl {
				builtins = append(builtins, k)
			}
		}
		sort.Strings(builtins)
		if len(builtins) > 0 {
			sb.WriteString(ui.SystemMsgStyle.Render("Built-in: "+strings.Join(builtins, ", ")) + "\n")
		}
		sb.WriteString("\n")

		sb.WriteString(ui.ActiveFlagStyle.Render("Your alias: "))
		if m.nickInput == "" {
			sb.WriteString(ui.HelpStyle.Render("_  (empty = clear existing)"))
		} else {
			sb.WriteString(lipgloss.NewStyle().Foreground(ui.ColorWhite).Bold(true).Render(m.nickInput + "_"))
		}
		sb.WriteString("\n\n")
		sb.WriteString(ui.HelpStyle.Render("Type alias · Enter to save · ESC to go back"))
	}

	return lipgloss.NewStyle().Height(m.viewportHeight()).Render(
		ui.BorderStyle.Width(m.width - 4).Render(sb.String()),
	)
}

func (m AppModel) viewModelPickerContent() string {
	var sb strings.Builder
	sb.WriteString(ui.TitleStyle.Render("Pick a Model") + "\n\n")

	if m.modelCustomMode {
		sb.WriteString(ui.ActiveFlagStyle.Render("Enter custom model name:") + "\n\n")
		sb.WriteString(fmt.Sprintf("  %s_\n\n", m.modelCustomInput))
		sb.WriteString(ui.HelpStyle.Render("Enter to confirm · ESC to go back"))
	} else {
		for i, mdl := range allModels {
			marker := "  "
			if i == m.modelCursor {
				marker = ui.ActiveFlagStyle.Render("▶ ")
			}
			numStr := ui.SystemMsgStyle.Render(fmt.Sprintf("[%2d]", i+1))
			var modelStr string
			if mdl == m.settings.NIMModel {
				modelStr = ui.TitleStyle.Render(mdl)
			} else {
				modelStr = lipgloss.NewStyle().Foreground(ui.ColorWhite).Render(mdl)
			}
			sb.WriteString(fmt.Sprintf("%s%s %s\n", marker, numStr, modelStr))
		}
		// Custom option.
		customMarker := "  "
		if m.modelCursor >= len(allModels) {
			customMarker = ui.ActiveFlagStyle.Render("▶ ")
		}
		sb.WriteString(fmt.Sprintf("%s%s %s\n",
			customMarker,
			ui.SystemMsgStyle.Render(fmt.Sprintf("[%2d]", len(allModels)+1)),
			ui.HelpStyle.Render("Custom (type any model name)"),
		))
		sb.WriteString("\n")
		sb.WriteString(ui.HelpStyle.Render("↑↓: navigate · Enter: select · ESC: back to settings"))
	}

	return lipgloss.NewStyle().Height(m.viewportHeight()).Render(
		ui.BorderStyle.Width(m.width - 4).Render(sb.String()),
	)
}

// ─── tea.Cmd factories ────────────────────────────────────────────────────────

func retrieveCmd(bridge *workers.Bridge, query string, k, window int, debug bool, minResults int, scoreThreshold float64) tea.Cmd {
	return func() (msg tea.Msg) {
		defer func() {
			if r := recover(); r != nil {
				crashLog("panic in retrieveCmd: %v\n%s", r, debug2.Stack())
				msg = RetrievalDoneMsg{Err: fmt.Errorf("panic in retrieval: %v", r)}
			}
		}()
		if bridge == nil {
			return RetrievalDoneMsg{Err: fmt.Errorf("bridge not initialised")}
		}
		results, ctx, stats, err := bridge.Retrieve(query, k, window, debug, minResults, scoreThreshold)
		return RetrievalDoneMsg{Results: results, Context: ctx, DebugStats: stats, Err: err}
	}
}

func waitForTokenCmd(tokenCh <-chan string, errCh <-chan error) tea.Cmd {
	return func() tea.Msg {
		// Block on tokenCh sequentially — no select race with errCh.
		// StreamLLM always closes tokenCh before sending to errCh, so once
		// we get ok=false we can safely drain errCh without the race.
		tok, ok := <-tokenCh
		if ok {
			return StreamTokenMsg{Token: tok}
		}
		// tokenCh closed — collect any pending error (nil = clean EOF).
		var err error
		select {
		case err = <-errCh:
		default:
		}
		return StreamDoneMsg{Err: err}
	}
}

func agentRetrievalCmd(bridge *workers.Bridge, query string, k, window int, debug bool, minResults int, scoreThreshold float64) tea.Cmd {
	return func() (msg tea.Msg) {
		defer func() {
			if r := recover(); r != nil {
				crashLog("panic in agentRetrievalCmd: %v\n%s", r, debug2.Stack())
				msg = AgentRetrievalDoneMsg{Query: query, Err: fmt.Errorf("panic in retrieval: %v", r)}
			}
		}()
		if bridge == nil {
			return AgentRetrievalDoneMsg{Query: query, Err: fmt.Errorf("bridge not initialised")}
		}
		results, chunkText, stats, err := bridge.Retrieve(query, k, window, debug, minResults, scoreThreshold)
		if err != nil {
			return AgentRetrievalDoneMsg{Query: query, Err: err}
		}
		topScore := 0.0
		if len(results) > 0 {
			topScore = results[0].RerankScore
			if topScore == 0 {
				topScore = results[0].Score
			}
		}
		return AgentRetrievalDoneMsg{
			Query:      query,
			NumResults: len(results),
			TopScore:   topScore,
			ChunkText:  chunkText,
			Results:    results,
			DebugStats: stats,
		}
	}
}

func fetchChatListCmd(bridge *workers.Bridge) tea.Cmd {
	return func() tea.Msg {
		if bridge == nil {
			return ChatListMsg{Err: fmt.Errorf("bridge not initialised")}
		}
		chats, active, err := bridge.ListChats()
		return ChatListMsg{Chats: chats, Active: active, Err: err}
	}
}

func fetchStatsCmd(bridge *workers.Bridge) tea.Cmd {
	return func() tea.Msg {
		if bridge == nil {
			return StatsMsg{Err: fmt.Errorf("bridge not initialised")}
		}
		stats, err := bridge.Stats()
		return StatsMsg{Stats: stats, Err: err}
	}
}

func fetchCheckCmd(bridge *workers.Bridge) tea.Cmd {
	return func() tea.Msg {
		if bridge == nil {
			return CheckResultMsg{Err: fmt.Errorf("bridge not initialised")}
		}
		data, err := bridge.CheckLatest()
		return CheckResultMsg{Data: data, Err: err}
	}
}

// ─── Pipeline cmd factories ───────────────────────────────────────────────────

// PipelineResultMsg carries the result of a pipeline operation.
type PipelineResultMsg struct {
	Action  string
	Message string
	Err     error
}

// isBridgePipeError returns true when err indicates the bridge subprocess has died.
func isBridgePipeError(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	return strings.Contains(s, "broken pipe") ||
		strings.Contains(s, "bridge stdout closed") ||
		strings.Contains(s, "write to bridge") ||
		strings.Contains(s, "read bridge response")
}

// restartBridgeCmd kills any existing bridge, starts a fresh one, and returns
// BridgeRestartedMsg on success or BridgeErrMsg on failure.
func restartBridgeCmd(ragDir string) tea.Cmd {
	return func() tea.Msg {
		b, err := workers.NewBridge(ragDir)
		if err != nil {
			return BridgeErrMsg{Err: fmt.Errorf("bridge restart failed: %w", err)}
		}
		return BridgeRestartedMsg{Bridge: b}
	}
}

// deriveSlugFromFilename converts a filename to the slug the bridge would assign.
// Mirrors bridge.py: stem → lowercase → replace non-alphanumeric/dash/underscore with '-'.
func deriveSlugFromFilename(filename string) string {
	base := filepath.Base(filename)
	if idx := strings.LastIndex(base, "."); idx > 0 {
		base = base[:idx]
	}
	var slug strings.Builder
	for _, ch := range strings.ToLower(base) {
		if (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' {
			slug.WriteRune(ch)
		} else {
			slug.WriteRune('-')
		}
	}
	result := strings.Trim(slug.String(), "-")
	if result == "" {
		return "default"
	}
	return result
}

func pipelineRebuildCmd(bridge *workers.Bridge) tea.Cmd {
	return func() tea.Msg {
		if bridge == nil {
			return PipelineResultMsg{Action: "rebuild", Err: fmt.Errorf("bridge not initialised")}
		}
		msg, err := bridge.Rebuild()
		return PipelineResultMsg{Action: "rebuild", Message: msg, Err: err}
	}
}

func pipelineTestCmd(bridge *workers.Bridge, query string, k, window int) tea.Cmd {
	return func() tea.Msg {
		if bridge == nil {
			return PipelineResultMsg{Action: "test", Err: fmt.Errorf("bridge not initialised")}
		}
		results, _, err := bridge.TestRetrieve(query, k)
		if err != nil {
			return PipelineResultMsg{Action: "test", Err: err}
		}
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("Test retrieve: %d results for %q\n", len(results), query))
		for i, r := range results {
			score := r.RerankScore
			if score == 0 {
				score = r.Score
			}
			text := r.Chunk.Text
			if len(text) > 70 {
				text = text[:70] + "…"
			}
			sb.WriteString(fmt.Sprintf("  [%d] %.4f  %s\n", i+1, score, text))
		}
		return PipelineResultMsg{Action: "test", Message: sb.String()}
	}
}

func pipelineIngestCmd(bridge *workers.Bridge, filePath string, limit int, afterDate string) tea.Cmd {
	progressCh := make(chan workers.IngestProgress, 20)
	return tea.Batch(
		// Goroutine: run ingest and stream progress into channel.
		func() tea.Msg {
			if bridge == nil {
				close(progressCh)
				return PipelineResultMsg{Action: "ingest", Err: fmt.Errorf("bridge not initialised")}
			}
			msg, err := bridge.IngestWithProgress(filePath, limit, afterDate, progressCh)
			return PipelineResultMsg{Action: "ingest", Message: msg, Err: err}
		},
		// Goroutine: relay first progress update; handler re-schedules for next.
		waitForIngestProgressCmd(progressCh),
	)
}

// waitForIngestProgressCmd reads from progressCh and re-schedules itself after
// each progress update, forming a self-perpetuating command chain.
func waitForIngestProgressCmd(ch chan workers.IngestProgress) tea.Cmd {
	return func() tea.Msg {
		p, ok := <-ch
		if !ok {
			return nil // channel closed — ingest done, stop the chain
		}
		return IngestProgressMsg{Pct: p.Pct, Message: p.Message, ch: ch}
	}
}

func switchChatCmd(bridge *workers.Bridge, slug string) tea.Cmd {
	return func() tea.Msg {
		if bridge == nil {
			return BridgeErrMsg{Err: fmt.Errorf("bridge not initialised")}
		}
		if err := bridge.SetChat(slug); err != nil {
			return BridgeErrMsg{Err: err}
		}
		chats, active, err := bridge.ListChats()
		return ChatListMsg{Chats: chats, Active: active, Err: err}
	}
}

// ─── Prompt building ──────────────────────────────────────────────────────────

func buildPrompt(query, retrievedContext, customTemplate, outputMode string) string {
	if customTemplate != "" {
		result := strings.ReplaceAll(customTemplate, "{query}", query)
		result = strings.ReplaceAll(result, "{context}", retrievedContext)
		return result
	}

	var schemaBlock string
	switch outputMode {
	case "rich":
		schemaBlock = richOutputSchema + clarifySchema
	case "structured":
		schemaBlock = structuredOutputSchema + clarifySchema
	default:
		schemaBlock = plainOutputRules + clarifySchema
	}

	return fmt.Sprintf(`You are an expert assistant. Answer the user's question using only the context provided below.

Question:
%s

Context chunks:
%s

%s`, query, retrievedContext, schemaBlock)
}

func buildAgentSystemPrompt(confident bool, maxSteps, searchesDone int, contextEnabled bool) string {
	var budgetLine string
	switch {
	case confident:
		budgetLine = "You MUST find a concrete answer. Do not give up early. Keep issuing SEARCH queries with different phrasings until you find direct evidence. Only emit ANSWER when you have found something specific, or have exhausted all reasonable search angles."
	case maxSteps > 0:
		remaining := maxSteps - searchesDone
		budgetLine = fmt.Sprintf("You have a budget of %d SEARCH queries remaining. Use them wisely — each should target something new and specific.", remaining)
	default:
		budgetLine = "You have no hard step limit, but be efficient. Only issue another SEARCH if you genuinely need more information; otherwise emit ANSWER to avoid wasting queries."
	}

	contextLine := "You will receive matched chunks plus surrounding context-window messages."
	if !contextEnabled {
		contextLine = "Surrounding context windows are disabled. You will only receive the matched chunk text."
	}

	return fmt.Sprintf(`You are querying a semantic search index built from Discord chat logs between two people: 'peepee' (Devin) and 'sania'.
SEARCH queries use semantic similarity — short, natural phrases (3-6 words) work best.
Each SEARCH returns the most relevant chat message chunks from the logs.
%s

Rules:
- Output exactly one line starting with SEARCH: <query> to retrieve chunks from the chat log
- Output exactly one response starting with ANSWER: <response> when you have enough information
- %s
- Use short, conversational phrases as queries — NOT boolean expressions or long sentences
- Each SEARCH must be meaningfully different from previous ones
- Do NOT repeat the same query twice
- If a search returns low scores, it means the data doesn't contain that phrasing — pivot completely
- NEVER build on a previous query by appending words to it
- If you've done 3+ searches with no good results, just ANSWER with what you found
- After 3-4 searches with no useful results, give your best ANSWER based on what was found
- Do NOT keep searching if you already have enough context to answer
- Do NOT output JSON, markdown, code fences, bullets, or explanations before SEARCH:/ANSWER:`, contextLine, budgetLine)
}

// parseAgentResponse scans the LLM response for the first SEARCH: or ANSWER:
// directive. It tolerates fenced blocks, labels like [SEARCH], and simple JSON.
func parseAgentResponse(response string) (action, payload string) {
	trimmed := strings.TrimSpace(response)
	if strings.HasPrefix(trimmed, "{") {
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(trimmed), &obj); err == nil {
			if a, ok := obj["action"].(string); ok {
				act := strings.ToUpper(strings.TrimSpace(a))
				switch act {
				case "SEARCH":
					for _, key := range []string{"query", "payload", "content"} {
						if v, ok := obj[key].(string); ok && strings.TrimSpace(v) != "" {
							return "SEARCH", strings.TrimSpace(v)
						}
					}
				case "ANSWER":
					for _, key := range []string{"answer", "payload", "content"} {
						if v, ok := obj[key].(string); ok && strings.TrimSpace(v) != "" {
							return "ANSWER", strings.TrimSpace(v)
						}
					}
				}
			}
		}
	}

	cleaned := strings.TrimSpace(trimmed)
	cleaned = regexp.MustCompile("(?s)<think>.*?</think>").ReplaceAllString(cleaned, "")
	cleaned = strings.ReplaceAll(cleaned, "```", "")

	re := regexp.MustCompile(`(?im)(?:^|[\n\r])\s*(?:[-*]\s*)?(?:\[\s*)?(SEARCH|ANSWER)(?:\s*\])?\s*:\s*(.+)`)
	match := re.FindStringSubmatch(cleaned)
	if match == nil {
		return "ANSWER", strings.TrimSpace(response)
	}

	foundAction := strings.ToUpper(strings.TrimSpace(match[1]))
	foundPayload := strings.TrimSpace(match[2])
	if foundAction == "SEARCH" {
		firstLine := strings.TrimSpace(strings.Split(foundPayload, "\n")[0])
		firstLine = strings.Trim(firstLine, `"'[] `)
		firstLine = regexp.MustCompile(`\s+(?:to|in order to)\b.*$`).ReplaceAllString(firstLine, "")
		return "SEARCH", firstLine
	}

	answerStart := strings.Index(strings.ToUpper(cleaned), "ANSWER:")
	if answerStart >= 0 {
		return "ANSWER", strings.TrimSpace(cleaned[answerStart+len("ANSWER:"):])
	}
	return "ANSWER", foundPayload
}

// ─── Utility ──────────────────────────────────────────────────────────────────

func (m AppModel) viewportHeight() int {
	acHeight := 0
	if m.acActive {
		n := len(m.acSuggestions) + 1 // +1 for hint line
		if n > 6 {
			n = 6
		}
		acHeight = n
	}
	clarifyHeight := 0
	if m.clarify.Active {
		clarifyHeight = m.clarify.PanelHeight()
	}
	h := m.height - 4 - acHeight - clarifyHeight - m.ingestBarRows()
	if h < 5 {
		return 5
	}
	return h
}

// ingestBarRows returns the number of rows occupied by the ingest progress bar.
func (m AppModel) ingestBarRows() int {
	if m.ingestInProgress {
		return 2
	}
	return 0
}

// helpVpHeight returns the inner height of the help pager viewport.
// The outer border box uses viewportHeight() rows total; the rounded border
// consumes 2 (top + bottom), and we also account for the 1-line hint above
// the input box when the help screen is active.
func (m AppModel) helpVpHeight() int {
	h := m.height - 4 - 1 - 2 - m.ingestBarRows() // total - inputRow(4) - border(2) - ingestBar
	if h < 3 {
		return 3
	}
	return h
}

// helpVpWidth returns the inner width of the help pager viewport.
// BorderStyle has border(2) + padding(1 each side = 2) = 4 chars overhead.
func (m AppModel) helpVpWidth() int {
	w := m.width - 4 - 4 // outer Width(m.width-4) minus border(2)+padding(2)
	if w < 20 {
		return 20
	}
	return w
}

func (m *AppModel) saveState() {
	path := filepath.Join(m.ragDir, ".tui_state.json")
	_ = SaveTUIState(path, m.tuiState)
}

func maskKey(key string) string {
	if len(key) <= 8 {
		return strings.Repeat("*", len(key))
	}
	return key[:4] + strings.Repeat("*", len(key)-8) + key[len(key)-4:]
}

// resolveModel maps a user-typed argument to a full model ID.
// Priority: number → user nickname → built-in nickname → substring match.
func (m AppModel) resolveModel(arg string) (string, bool) {
	a := strings.TrimSpace(arg)
	if a == "" {
		return "", false
	}
	// 1. Number.
	if n, err := strconv.Atoi(a); err == nil {
		if n >= 1 && n <= len(allModels) {
			return allModels[n-1], true
		}
		return "", false
	}
	low := strings.ToLower(a)
	// 2. User nickname (exact).
	if mdl, ok := m.modelNicknames[low]; ok {
		return mdl, true
	}
	// 3. Built-in nickname (exact).
	if mdl, ok := defaultNicknames[low]; ok {
		return mdl, true
	}
	// 4. Exact full model ID.
	for _, mdl := range allModels {
		if strings.EqualFold(mdl, a) {
			return mdl, true
		}
	}
	// 5. Substring match against model IDs.
	var subMatches []string
	for _, mdl := range allModels {
		if strings.Contains(strings.ToLower(mdl), low) {
			subMatches = append(subMatches, mdl)
		}
	}
	if len(subMatches) == 1 {
		return subMatches[0], true
	}
	// Custom model not in the catalogue (user typed a full path).
	if strings.ContainsRune(a, '/') {
		return a, true
	}
	return "", false
}

// nicknamesFor returns all aliases (user + built-in) that point to modelID.
func (m AppModel) nicknamesFor(modelID string) []string {
	var out []string
	for k, v := range defaultNicknames {
		if v == modelID {
			out = append(out, k)
		}
	}
	for k, v := range m.modelNicknames {
		if v == modelID {
			out = append(out, "*"+k) // * marks user-defined
		}
	}
	sort.Strings(out)
	return out
}

// saveNicknames writes m.modelNicknames to disk.
func (m AppModel) saveNicknames() error {
	data, err := json.MarshalIndent(m.modelNicknames, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(m.nicknamePath, data, 0644)
}

// loadNicknameFile reads a user nickname file from disk.
// Returns an empty map on any error or missing file.
func loadNicknameFile(path string) map[string]string {
	out := make(map[string]string)
	data, err := os.ReadFile(path)
	if err != nil {
		return out
	}
	_ = json.Unmarshal(data, &out)
	return out
}

func sortedSlugs(chats map[string]interface{}) []string {
	slugs := make([]string, 0, len(chats))
	for k := range chats {
		slugs = append(slugs, k)
	}
	// Simple insertion sort for deterministic ordering.
	for i := 1; i < len(slugs); i++ {
		for j := i; j > 0 && slugs[j] < slugs[j-1]; j-- {
			slugs[j], slugs[j-1] = slugs[j-1], slugs[j]
		}
	}
	return slugs
}

// ─── Chat file list screen ────────────────────────────────────────────────────

func loadChatFileListCmd(ragDir string) tea.Cmd {
	return func() tea.Msg {
		dir := filepath.Join(ragDir, "raw")
		entries, err := os.ReadDir(dir)
		if err != nil {
			return ChatFileListMsg{Err: fmt.Errorf("raw/ dir not found: %w", err)}
		}
		var files []string
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			name := e.Name()
			if strings.HasSuffix(name, ".json") || strings.HasSuffix(name, ".csv") {
				files = append(files, name)
			}
		}
		sort.Strings(files)
		return ChatFileListMsg{Files: files}
	}
}

func loadChatFileCmd(ragDir, filename string) tea.Cmd {
	return func() tea.Msg {
		path := filepath.Join(ragDir, "raw", filepath.Base(filename))
		data, err := os.ReadFile(path)
		if err != nil {
			return ChatFileLoadedMsg{Err: err}
		}
		msgs, err := parseChatJSON(data)
		if err != nil {
			return ChatFileLoadedMsg{Err: fmt.Errorf("parse: %w", err)}
		}
		const maxMsgs = 5000
		truncated := false
		if len(msgs) > maxMsgs {
			msgs = msgs[:maxMsgs]
			truncated = true
		}
		content := formatChatForViewer(msgs)
		if truncated {
			content += "\n" + ui.HelpStyle.Render(fmt.Sprintf("  (showing first %d messages)", maxMsgs))
		}
		return ChatFileLoadedMsg{Title: filename, Content: content}
	}
}

// parseChatJSON normalizes any chat JSON format into ViewerMessages.
func parseChatJSON(data []byte) ([]workers.ViewerMessage, error) {
	extractMsg := func(m map[string]interface{}) (workers.ViewerMessage, bool) {
		var sender string
		// Try each field as string or as nested object containing a name.
		for _, k := range []string{"sender", "author", "from", "user", "member", "poster"} {
			if sender != "" {
				break
			}
			if v, ok := m[k].(string); ok && v != "" {
				sender = v
				break
			}
			if obj, ok := m[k].(map[string]interface{}); ok {
				for _, nk := range []string{"nickname", "display_name", "name", "username",
					"real_name", "realName", "from_name", "user_name", "title"} {
					if v, ok := obj[nk].(string); ok && v != "" {
						sender = v
						break
					}
				}
			}
		}
		text, _ := m["text"].(string)
		for _, k := range []string{"content", "message", "body"} {
			if text == "" {
				text, _ = m[k].(string)
			}
		}
		// Also handle text as []interface{} (Telegram-style rich text array).
		if text == "" {
			if arr, ok := m["text"].([]interface{}); ok {
				var parts []string
				for _, part := range arr {
					switch v := part.(type) {
					case string:
						parts = append(parts, v)
					case map[string]interface{}:
						if t, ok := v["text"].(string); ok {
							parts = append(parts, t)
						}
					}
				}
				text = strings.Join(parts, "")
			}
		}
		ts, _ := m["timestamp"].(string)
		for _, k := range []string{"date", "time", "created_at", "ts", "date_unixtime"} {
			if ts == "" {
				ts, _ = m[k].(string)
			}
		}
		// Convert Slack-style unix epoch string (e.g. "1617000000.000000") to ISO.
		if ts != "" {
			if f, err := strconv.ParseFloat(ts, 64); err == nil && f > 1_000_000_000 {
				ts = time.Unix(int64(f), 0).UTC().Format("2006-01-02T15:04:05")
			}
		}
		if text == "" {
			return workers.ViewerMessage{}, false
		}
		if sender == "" {
			sender = "(unknown)"
		}
		return workers.ViewerMessage{Sender: sender, Text: text, Timestamp: ts}, true
	}

	toMsgs := func(items []interface{}) []workers.ViewerMessage {
		var out []workers.ViewerMessage
		for _, item := range items {
			if m, ok := item.(map[string]interface{}); ok {
				if msg, ok := extractMsg(m); ok {
					out = append(out, msg)
				}
			}
		}
		return out
	}

	// Try direct array.
	var arr []interface{}
	if json.Unmarshal(data, &arr) == nil {
		return toMsgs(arr), nil
	}

	// Try object with known wrapper keys.
	var obj map[string]interface{}
	if err := json.Unmarshal(data, &obj); err != nil {
		return nil, err
	}
	for _, key := range []string{"messages", "msgs", "chats", "data"} {
		if v, ok := obj[key]; ok {
			if items, ok := v.([]interface{}); ok {
				return toMsgs(items), nil
			}
		}
	}
	// Fall back to first array value in the object.
	for _, v := range obj {
		if items, ok := v.([]interface{}); ok {
			return toMsgs(items), nil
		}
	}
	return nil, fmt.Errorf("no message array found in JSON")
}

// formatChatForViewer renders a slice of ViewerMessages into styled text for the viewport.
func formatChatForViewer(msgs []workers.ViewerMessage) string {
	if len(msgs) == 0 {
		return ui.SystemMsgStyle.Render("  (no messages found in this file)")
	}

	dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
	senderStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
	dayStyle := lipgloss.NewStyle().Foreground(ui.ColorGreen).Bold(true)
	textStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite)

	var sb strings.Builder
	currentDay := ""

	for _, m := range msgs {
		ts := m.Timestamp
		day := ""
		if len(ts) >= 10 {
			day = ts[:10]
		}
		if day != "" && day != currentDay {
			currentDay = day
			sb.WriteString("\n" + dayStyle.Render("  ── "+day+" ──") + "\n\n")
		}

		timeStr := ""
		if len(ts) >= 16 {
			timeStr = ts[11:16]
		} else if len(ts) > 10 {
			timeStr = ts[10:]
		}

		sender := m.Sender
		if len(sender) > 18 {
			sender = sender[:18]
		}

		timeRendered := dimStyle.Render(fmt.Sprintf("[%s]", timeStr))
		senderRendered := senderStyle.Render(fmt.Sprintf("%-18s", sender))

		// Split on newlines in the message itself.
		lines := strings.Split(strings.TrimRight(m.Text, "\n"), "\n")
		indent := strings.Repeat(" ", 2+7+2+18+2) // matches prefix width
		for i, line := range lines {
			if i == 0 {
				sb.WriteString("  " + timeRendered + "  " + senderRendered + "  " + textStyle.Render(line) + "\n")
			} else if strings.TrimSpace(line) != "" {
				sb.WriteString(indent + textStyle.Render(line) + "\n")
			}
		}
	}

	return sb.String()
}

func (m *AppModel) handleChatListKey(msg tea.KeyMsg) []tea.Cmd {
	m.acConsumedKey = true
	switch msg.Type {
	case tea.KeyEsc:
		m.screen = ScreenSettingsMenu
	case tea.KeyUp:
		if m.chatFileCursor > 0 {
			m.chatFileCursor--
		}
	case tea.KeyDown:
		if m.chatFileCursor < len(m.chatFileList)-1 {
			m.chatFileCursor++
		}
	case tea.KeyEnter:
		if len(m.chatFileList) > 0 {
			filename := m.chatFileList[m.chatFileCursor]
			return []tea.Cmd{loadChatFileCmd(m.ragDir, filename)}
		}
	}
	return nil
}

func (m AppModel) viewChatListContent() string {
	var sb strings.Builder
	sb.WriteString(ui.TitleStyle.Render("Browse Raw Chat Files") + "\n\n")

	if len(m.chatFileList) == 0 {
		sb.WriteString(ui.SystemMsgStyle.Render("  Loading…") + "\n")
	} else {
		for i, f := range m.chatFileList {
			isSelected := i == m.chatFileCursor
			marker := "  "
			label := f
			if isSelected {
				marker = ui.ActiveFlagStyle.Render("▶ ")
				label = lipgloss.NewStyle().Foreground(ui.ColorWhite).Render(f)
			} else {
				label = ui.SystemMsgStyle.Render(f)
			}
			sb.WriteString(fmt.Sprintf("%s%s\n", marker, label))
		}
	}
	sb.WriteString("\n")
	sb.WriteString(ui.HelpStyle.Render("↑↓: navigate · Enter: open · ESC: back"))
	return lipgloss.NewStyle().Height(m.viewportHeight()).Render(
		ui.BorderStyle.Width(m.width - 4).Render(sb.String()),
	)
}

// ─── Chat viewer screen ───────────────────────────────────────────────────────

func (m *AppModel) handleChatViewerKey(msg tea.KeyMsg) []tea.Cmd {
	m.acConsumedKey = true

	// Search mode: route keystrokes to search input.
	if m.chatViewSearchMode {
		switch msg.Type {
		case tea.KeyEsc:
			m.chatViewSearchMode = false
			m.chatViewSearchTerm = ""
			m.chatViewMatchLines = nil
			m.chatViewVP.Height = m.chatViewVpHeight()
		case tea.KeyEnter:
			if m.chatViewSearchTerm != "" {
				m.viewerNextMatch()
			} else {
				m.chatViewSearchMode = false
				m.chatViewVP.Height = m.chatViewVpHeight()
			}
		case tea.KeyBackspace, tea.KeyDelete:
			if len(m.chatViewSearchTerm) > 0 {
				m.chatViewSearchTerm = m.chatViewSearchTerm[:len(m.chatViewSearchTerm)-1]
				m.viewerDoSearch()
			}
		default:
			if r := msg.Runes; len(r) > 0 {
				m.chatViewSearchTerm += string(r)
				m.viewerDoSearch()
			}
		}
		return nil
	}

	switch msg.Type {
	case tea.KeyEsc:
		m.screen = ScreenChatList
	case tea.KeyUp:
		m.chatViewVP.LineUp(1)
	case tea.KeyDown:
		m.chatViewVP.LineDown(1)
	case tea.KeyPgUp:
		m.chatViewVP.HalfViewUp()
	case tea.KeyPgDown:
		m.chatViewVP.HalfViewDown()
	case tea.KeyHome:
		m.chatViewVP.GotoTop()
	case tea.KeyEnd:
		m.chatViewVP.GotoBottom()
	default:
		switch msg.String() {
		case "/", "f":
			m.chatViewSearchMode = true
			m.chatViewSearchTerm = ""
			m.chatViewMatchLines = nil
			m.chatViewVP.Height = m.chatViewVpHeight()
		case "n":
			m.viewerNextMatch()
		case "N":
			m.viewerPrevMatch()
		}
	}
	return nil
}

var ansiStripRe = regexp.MustCompile(`\x1b\[[0-9;]*m`)
var browserLeadingTimestampRe = regexp.MustCompile(`^\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?\]?\s*`)

func (m *AppModel) viewerDoSearch() {
	term := strings.ToLower(m.chatViewSearchTerm)
	if term == "" {
		m.chatViewMatchLines = nil
		return
	}
	stripped := ansiStripRe.ReplaceAllString(m.chatViewContent, "")
	lines := strings.Split(stripped, "\n")
	var matches []int
	for i, line := range lines {
		if strings.Contains(strings.ToLower(line), term) {
			matches = append(matches, i)
		}
	}
	m.chatViewMatchLines = matches
	m.chatViewMatchIdx = 0
	if len(matches) > 0 {
		m.viewerScrollToMatch(matches[0])
	}
}

func (m *AppModel) viewerScrollToMatch(lineIdx int) {
	m.chatViewVP.GotoTop()
	if lineIdx > 0 {
		m.chatViewVP.LineDown(lineIdx)
	}
}

func (m *AppModel) viewerNextMatch() {
	if len(m.chatViewMatchLines) == 0 {
		return
	}
	m.chatViewMatchIdx = (m.chatViewMatchIdx + 1) % len(m.chatViewMatchLines)
	m.viewerScrollToMatch(m.chatViewMatchLines[m.chatViewMatchIdx])
}

func (m *AppModel) viewerPrevMatch() {
	if len(m.chatViewMatchLines) == 0 {
		return
	}
	m.chatViewMatchIdx = (m.chatViewMatchIdx - 1 + len(m.chatViewMatchLines)) % len(m.chatViewMatchLines)
	m.viewerScrollToMatch(m.chatViewMatchLines[m.chatViewMatchIdx])
}

func (m AppModel) viewChatViewerContent() string {
	pct := 0
	if m.chatViewVP.TotalLineCount() > 0 {
		pct = int(m.chatViewVP.ScrollPercent() * 100)
	}

	title := ui.TitleStyle.Render(m.chatViewTitle)
	scroll := ui.HelpStyle.Render(fmt.Sprintf("%d%%", pct))

	// Match count badge when search has results.
	matchInfo := ""
	if m.chatViewSearchTerm != "" && len(m.chatViewMatchLines) > 0 {
		matchInfo = " " + ui.ActiveFlagStyle.Render(fmt.Sprintf("%d/%d", m.chatViewMatchIdx+1, len(m.chatViewMatchLines)))
	} else if m.chatViewSearchTerm != "" && len(m.chatViewMatchLines) == 0 {
		matchInfo = " " + ui.ErrorMsgStyle.Render("no match")
	}

	header := lipgloss.JoinHorizontal(lipgloss.Top, title, "  ", scroll, matchInfo)

	// Search bar line shown when search mode is active.
	var searchLine string
	if m.chatViewSearchMode {
		prompt := ui.SystemMsgStyle.Render("/")
		cursor := "█"
		searchLine = "\n" + prompt + m.chatViewSearchTerm + cursor +
			ui.HelpStyle.Render("  Enter: next · n/N: next/prev · ESC: close")
	}

	helpText := ui.HelpStyle.Render("↑↓ / PgUp/PgDn: scroll · Home/End · /: find · ESC: back")
	inner := header + "\n\n" + m.chatViewVP.View() + "\n" + helpText + searchLine

	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(ui.ColorCyan).
		Padding(0, 1).
		Width(m.width - 2).
		Render(inner)
}

func (m AppModel) chatViewVpHeight() int {
	// Layout: border(2) + header(1) + blank(1) + VP + blank(1) + help(1) + inputRow(3) + statusBar(1) = 10
	extra := 0
	if m.chatViewSearchMode {
		extra = 1 // search bar takes one extra line
	}
	h := m.height - 10 - extra
	if h < 3 {
		return 3
	}
	return h
}

func (m AppModel) chatViewVpWidth() int {
	w := m.width - 2 - 4 // outer Width(m.width-2) minus border(2)+padding(2)
	if w < 20 {
		return 20
	}
	return w
}

// ─── RAG browser ──────────────────────────────────────────────────────────────

func (m *AppModel) openRagBrowser() {
	if len(m.ragBrowseChunks) == 0 {
		return
	}
	m.ragBrowseCursor = 0
	m.ragBrowseCtxOpen = false
	m.resizeRagBrowserViewports()
	m.ragBrowseVP.GotoTop()
	m.screen = ScreenRagBrowser
}

func (m *AppModel) resizeRagBrowserViewports() {
	m.ragBrowseVP.Width = m.ragBrowserInnerWidth()
	m.ragBrowseVP.Height = m.ragBrowserViewportHeight()
	m.ragBrowseVP.SetContent(m.renderRagBrowserList())
	m.ensureRagBrowseCursorVisible()

	m.ragBrowseCtxVP.Width = m.ragBrowserInnerWidth()
	m.ragBrowseCtxVP.Height = m.ragBrowserViewportHeight()
	if m.ragBrowseCtxOpen {
		m.ragBrowseCtxVP.SetContent(m.renderRagBrowserCtx())
	}
}

func (m AppModel) ragBrowseCardHeight(r workers.Result) int {
	return lipgloss.Height(m.renderRagBrowserCard(r, -1))
}

func (m AppModel) ragBrowseCardOffset(idx int) int {
	offset := 0
	for i := 0; i < idx && i < len(m.ragBrowseChunks); i++ {
		offset += lipgloss.Height(m.renderRagBrowserCard(m.ragBrowseChunks[i], i)) + 1
	}
	return offset
}

func (m *AppModel) ensureRagBrowseCursorVisible() {
	if m.ragBrowseCursor < 0 || m.ragBrowseCursor >= len(m.ragBrowseChunks) {
		return
	}
	top := m.ragBrowseCardOffset(m.ragBrowseCursor)
	bottom := top + lipgloss.Height(m.renderRagBrowserCard(m.ragBrowseChunks[m.ragBrowseCursor], m.ragBrowseCursor))
	if top < m.ragBrowseVP.YOffset {
		m.ragBrowseVP.YOffset = top
	} else if bottom > m.ragBrowseVP.YOffset+m.ragBrowseVP.Height {
		m.ragBrowseVP.YOffset = bottom - m.ragBrowseVP.Height
	}
	if m.ragBrowseVP.YOffset < 0 {
		m.ragBrowseVP.YOffset = 0
	}
}

func (m *AppModel) renderRagBrowserList() string {
	var parts []string
	for i, r := range m.ragBrowseChunks {
		if i > 0 {
			parts = append(parts, "")
		}
		parts = append(parts, m.renderRagBrowserCard(r, i))
	}
	return strings.Join(parts, "\n")
}

func (m *AppModel) renderRagBrowserCard(r workers.Result, idx int) string {
	accentStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
	dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
	rankStyle := lipgloss.NewStyle().Foreground(ui.ColorGreen).Bold(true)
	scoreStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan)
	starStyle := lipgloss.NewStyle().Foreground(ui.ColorYellow).Bold(true)
	srcStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite).Bold(true)
	textStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite)

	cardWidth := m.ragBrowserInnerWidth() - 4 // subtract border(2) + padding(2) overhead
	innerWidth := max(16, cardWidth-4)

	score := r.RerankScore
	if score == 0 {
		score = r.Score
	}
	const barLen = 10
	filled := clampScoreBarFill(score, barLen)
	bar := scoreStyle.Render(strings.Repeat("█", filled)) +
		dimStyle.Render(strings.Repeat("░", barLen-filled))

	star := dimStyle.Render("·")
	if r.KeywordBoosted {
		star = starStyle.Render("★")
	}
	meta := rankStyle.Render(fmt.Sprintf("#%-2d", idx+1)) + "  " +
		bar + "  " +
		scoreStyle.Render(fmt.Sprintf("%.4f", score)) + "  " +
		star + "  " +
		srcStyle.Render(r.Source) + "  " +
		dimStyle.Render(browserDisplayTimestamp(r.Chunk.TimestampStart))

	text := sanitizeBrowserChunkText(r.Chunk.Text, r.Chunk.Sender)
	maxLen := innerWidth * 4
	if maxLen < 80 {
		maxLen = 80
	}
	if len(text) > maxLen {
		text = text[:maxLen] + "…"
	}
	lines := wrapText(text, innerWidth)
	if len(lines) == 0 {
		lines = []string{"(empty chunk)"}
	}

	sep := dimStyle.Render(strings.Repeat("─", innerWidth))
	body := meta + "\n" + sep + "\n" + textStyle.Render(strings.Join(lines, "\n"))

	borderColor := ui.ColorDim
	if idx == m.ragBrowseCursor {
		borderColor = ui.ColorCyan
		body = accentStyle.Render("▶ ") + body
	}

	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(borderColor).
		Padding(0, 1).
		Width(cardWidth).
		Render(body)
}

func (m *AppModel) renderRagBrowserCtx() string {
	if m.ragBrowseCursor >= len(m.ragBrowseChunks) {
		return ""
	}
	r := m.ragBrowseChunks[m.ragBrowseCursor]

	titleStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
	dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
	userStyle := lipgloss.NewStyle().Foreground(ui.ColorGreen).Bold(true)
	tsStyle := lipgloss.NewStyle().Foreground(ui.ColorYellow)
	textStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite)

	cardWidth := m.ragBrowserInnerWidth() - 4 // subtract card border(2) + padding(2) overhead
	innerWidth := max(20, cardWidth-4)

	var lines []string
	title := titleStyle.Render(fmt.Sprintf("Context window — chunk #%d", m.ragBrowseCursor+1))
	lines = append(lines, title)
	lines = append(lines, dimStyle.Render(strings.Repeat("─", innerWidth)))

	context := r.ContextWindow
	if len(context) == 0 {
		context = []workers.Chunk{r.Chunk}
	}
	anchorIdx := len(context) / 2
	if len(context) > 0 {
		for i, chunk := range context {
			if chunk.ChunkID == r.Chunk.ChunkID {
				anchorIdx = i
				break
			}
		}
	}

	senderWidth := browserSenderWidth(append([]workers.Chunk{r.Chunk}, context...))
	textWidth := max(20, innerWidth-16-2-senderWidth-2)
	before := browserTimelineRows(context[:anchorIdx], senderWidth, textWidth, userStyle, tsStyle, textStyle)
	after := browserTimelineRows(context[min(anchorIdx+1, len(context)):], senderWidth, textWidth, userStyle, tsStyle, textStyle)

	if len(before) > 0 {
		lines = append(lines, dimStyle.Render("Earlier context"))
		lines = append(lines, before...)
		lines = append(lines, "")
	}

	lines = append(lines, renderBrowserMainMatchCard(r.Chunk, cardWidth))

	if len(after) > 0 {
		lines = append(lines, "")
		lines = append(lines, dimStyle.Render("Later context"))
		lines = append(lines, after...)
	} else if len(before) == 0 {
		lines = append(lines, "")
		lines = append(lines, dimStyle.Render("(no surrounding context available)"))
	}
	return strings.Join(lines, "\n")
}

func clampScoreBarFill(score float64, barLen int) int {
	filled := int(score * float64(barLen))
	if filled < 0 {
		return 0
	}
	if filled > barLen {
		return barLen
	}
	return filled
}

func sanitizeBrowserChunkText(text, sender string) string {
	clean := strings.TrimSpace(strings.Join(strings.Fields(text), " "))
	clean = browserLeadingTimestampRe.ReplaceAllString(clean, "")
	if sender != "" {
		senderPrefixRe := regexp.MustCompile(`^` + regexp.QuoteMeta(sender) + `\s*:\s*`)
		clean = senderPrefixRe.ReplaceAllString(clean, "")
	}
	return strings.TrimSpace(clean)
}

func browserDisplayTimestamp(ts string) string {
	ts = strings.TrimSpace(strings.ReplaceAll(ts, "T", " "))
	if len(ts) > 16 {
		ts = ts[:16]
	}
	return ts
}

func browserSenderWidth(chunks []workers.Chunk) int {
	width := 8
	for _, chunk := range chunks {
		if w := lipgloss.Width(chunk.Sender); w > width {
			width = w
		}
	}
	if width > 14 {
		width = 14
	}
	return width
}

func browserTimelineRows(chunks []workers.Chunk, senderWidth, textWidth int, senderStyle, tsStyle, textStyle lipgloss.Style) []string {
	var rows []string
	for _, chunk := range chunks {
		sender := chunk.Sender
		if sender == "" {
			sender = "(unknown)"
		}
		ts := browserDisplayTimestamp(chunk.TimestampStart)
		body := sanitizeBrowserChunkText(chunk.Text, chunk.Sender)
		wrapped := wrapText(body, textWidth)
		if len(wrapped) == 0 {
			wrapped = []string{"(empty chunk)"}
		}
		prefixPlain := fmt.Sprintf("%-16s  %-*s  ", ts, senderWidth, sender)
		prefix := tsStyle.Render(fmt.Sprintf("%-16s", ts)) + "  " +
			senderStyle.Render(fmt.Sprintf("%-*s", senderWidth, sender)) + "  "
		rows = append(rows, prefix+textStyle.Render(wrapped[0]))
		indent := strings.Repeat(" ", lipgloss.Width(prefixPlain))
		for _, line := range wrapped[1:] {
			rows = append(rows, indent+textStyle.Render(line))
		}
	}
	return rows
}

func renderBrowserMainMatchCard(chunk workers.Chunk, width int) string {
	titleStyle := lipgloss.NewStyle().Foreground(ui.ColorYellow).Bold(true)
	dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
	userStyle := lipgloss.NewStyle().Foreground(ui.ColorGreen).Bold(true)
	textStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite)

	bodyWidth := max(20, width-6)
	lines := wrapText(sanitizeBrowserChunkText(chunk.Text, chunk.Sender), bodyWidth)
	if len(lines) == 0 {
		lines = []string{"(empty chunk)"}
	}

	content := titleStyle.Render("Main match") + "\n" +
		dimStyle.Render(browserDisplayTimestamp(chunk.TimestampStart)) + "  " +
		userStyle.Render(chunk.Sender) + "\n\n" +
		textStyle.Render(strings.Join(lines, "\n"))

	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(ui.ColorYellow).
		Padding(0, 1).
		Width(width).
		Render(content)
}

func (m *AppModel) handleRagBrowserKey(msg tea.KeyMsg) []tea.Cmd {
	m.acConsumedKey = true
	switch msg.Type {
	case tea.KeyEsc:
		if m.ragBrowseCtxOpen {
			m.ragBrowseCtxOpen = false
		} else {
			m.screen = ScreenChat
		}
	case tea.KeyUp:
		if m.ragBrowseCtxOpen {
			m.ragBrowseCtxVP.LineUp(3)
		} else {
			if m.ragBrowseCursor > 0 {
				m.ragBrowseCursor--
				m.ragBrowseVP.SetContent(m.renderRagBrowserList())
				m.ensureRagBrowseCursorVisible()
			}
		}
	case tea.KeyDown:
		if m.ragBrowseCtxOpen {
			m.ragBrowseCtxVP.LineDown(3)
		} else {
			if m.ragBrowseCursor < len(m.ragBrowseChunks)-1 {
				m.ragBrowseCursor++
				m.ragBrowseVP.SetContent(m.renderRagBrowserList())
				m.ensureRagBrowseCursorVisible()
			}
		}
	case tea.KeyPgUp:
		if m.ragBrowseCtxOpen {
			m.ragBrowseCtxVP.HalfViewUp()
		} else {
			m.ragBrowseVP.HalfViewUp()
		}
	case tea.KeyPgDown:
		if m.ragBrowseCtxOpen {
			m.ragBrowseCtxVP.HalfViewDown()
		} else {
			m.ragBrowseVP.HalfViewDown()
		}
	case tea.KeyEnter:
		if !m.ragBrowseCtxOpen && m.ragBrowseCursor < len(m.ragBrowseChunks) {
			m.ragBrowseCtxVP = viewport.New(m.ragBrowserInnerWidth(), m.ragBrowserViewportHeight())
			m.ragBrowseCtxVP.SetContent(m.renderRagBrowserCtx())
			m.ragBrowseCtxVP.GotoTop()
			m.ragBrowseCtxOpen = true
		}
	}
	return nil
}

func (m AppModel) viewRagBrowserContent() string {
	dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
	titleStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)

	n := len(m.ragBrowseChunks)
	title := titleStyle.Render(fmt.Sprintf("Chunk Browser  (%d/%d)", m.ragBrowseCursor+1, n))
	help := dimStyle.Render("↑↓ move · Enter view context · Esc back")

	var inner string
	if m.ragBrowseCtxOpen {
		ctxTitle := titleStyle.Render(fmt.Sprintf("Context — chunk #%d", m.ragBrowseCursor+1))
		ctxHelp := dimStyle.Render("↑↓ / PgUp/PgDn scroll · Esc back to list")
		inner = ctxTitle + "\n" + m.ragBrowseCtxVP.View() + "\n" + ctxHelp
	} else {
		inner = title + "\n" + m.ragBrowseVP.View() + "\n" + help
	}

	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(ui.ColorCyan).
		Padding(0, 1).
		Width(m.ragBrowserBoxWidth()).
		Height(m.viewportHeight()).
		Render(inner)
}

func (m AppModel) ragBrowserBoxWidth() int {
	w := m.width - 4
	if w < 24 {
		return 24
	}
	return w
}

func (m AppModel) ragBrowserInnerWidth() int {
	w := m.ragBrowserBoxWidth() - 4
	if w < 20 {
		return 20
	}
	return w
}

func (m AppModel) ragBrowserViewportHeight() int {
	h := m.viewportHeight() - 4
	if h < 3 {
		return 3
	}
	return h
}
