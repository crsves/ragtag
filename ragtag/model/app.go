package model

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
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

// AgentLLMDoneMsg carries the parsed action from one agentic LLM call.
type AgentLLMDoneMsg struct {
	Action  string // "SEARCH" or "ANSWER"
	Payload string
}

// AgentRetrievalDoneMsg carries retrieval results for one agentic search step.
type AgentRetrievalDoneMsg struct {
	Query      string
	NumResults int
	TopScore   float64
	ChunkText  string
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
	{"/pause", "pause/stop current AI response", 3},
	{"/model", "set model by name/number/nickname", 4},
	{"/k", "set final_k (chunks passed to LLM)", 5},
	{"/window", "set context window size", 6},
	{"/settings", "open settings panel", 7},
	{"/chats", "switch between chats", 8},
	{"/help", "show command reference", 9},
	{"/stats", "show index statistics", 10},
	{"/pipeline", "open pipeline management", 10},
	{"/view", "browse raw chat history", 10},
	{"/confident", "toggle confident mode", 11},
	{"/thinking", "toggle thinking mode", 12},
	{"/rag", "toggle RAG-only mode (skip LLM)", 13},
	{"/minresults", "set minimum results floor (e.g. /minresults 5)", 13},
	{"/threshold", "set score threshold (e.g. /threshold 1.5)", 13},
	{"/back", "go back to chat", 13},
	{"/exit", "exit the TUI", 14},
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
	"deepseek":  "deepseek-ai/deepseek-v3.2",
	"gpt":       "openai/gpt-oss-120b",
	"gpt120":    "openai/gpt-oss-120b",
	"llama":     "meta/llama-3.3-70b-instruct",
	"llama70":   "meta/llama-3.3-70b-instruct",
	"llama8":    "meta/llama-3.1-8b-instruct",
	"llama3b":   "meta/llama-3.2-3b-instruct",
	"nemotron":  "nvidia/llama-3.1-nemotron-70b-instruct",
	"mistral":   "mistralai/mistral-7b-instruct-v0.3",
	"mixtral":   "mistralai/mixtral-8x7b-instruct-v0.1",
	"gemma":     "google/gemma-2-9b-it",
	"phi":       "microsoft/phi-3-mini-128k-instruct",
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
	{"Rebuild full index from raw data", "rebuild"},
	{"Test retrieval (no LLM call)", "test"},
	{"Show index stats", "stats"},
	{"Ingest new data (incremental)", "ingest"},
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
	appState AppState
	screen   Screen

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
	agentCtx        bytes.Buffer    // accumulated retrieved context (not the Go context.Context)
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
	errMsg      string
	bridgeReady bool
	mdRenderer  *glamour.TermRenderer

	// Autocomplete
	acSuggestions []cmdSuggestion
	acSelected    int
	acActive      bool
	acConsumedKey bool // true = skip textarea/viewport Update for this msg

	// Help pager
	helpVp viewport.Model

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
	pipelineCursor    int
	pipelineInputMode bool   // true = capturing text input (query / file path)
	pipelineInput     string // text typed in pipeline input mode
	pipelinePrompt    string // what we're asking for ("query" or "file")

	// Chat file viewer
	chatFileList      []string
	chatFileCursor    int
	chatViewVP        viewport.Model
	chatViewTitle     string
	chatViewContent   string // rendered content stored for search
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
	nickStep     int    // 0 = pick model, 1 = type name
	nickModelIdx int
	nickInput    string

	// Startup animation
	animFrame int
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
	if m.bridge == nil {
		return tea.Batch(m.spinner.Tick, animTickCmd())
	}
	return tea.Batch(m.spinner.Tick, animTickCmd(), waitForBridgeCmd(m.bridge))
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
		m.helpVp.SetContent(m.buildHelpContent())
		m.chatViewVP.Width = m.chatViewVpWidth()
		m.chatViewVP.Height = m.chatViewVpHeight()
		m.renderMessages()

	// ── Spinner tick ───────────────────────────────────────────────────────
	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		if m.appState != StateIdle {
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
			m.addMessage(ChatMessage{Role: "system", Content: "debug · retrieval stats\n" + sb.String()})

			// Results table — in debug mode show up to 10 with full detail.
			limit := len(msg.Results)
			if limit > 10 {
				limit = 10
			}
			if limit > 0 {
				var rb strings.Builder
				rb.WriteString(dimStyle.Render(fmt.Sprintf("  %-3s  %-7s  %-3s  %-8s  %-19s  %s", "#", "score", "★", "src", "timestamp", "text")) + "\n")
				rb.WriteString(dimStyle.Render("  " + strings.Repeat("─", 80)) + "\n")
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
				m.addMessage(ChatMessage{Role: "system", Content: "debug · top results\n" + rb.String()})
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
			finalMsg := ChatMessage{
				Role:    "assistant",
				Content: content,
				Sources: m.streamSources,
			}
			m.addMessage(finalMsg)
		}
		m.streamSources = nil
		if msg.Err != nil {
			m.addMessage(ChatMessage{Role: "error", Content: "Stream error: " + msg.Err.Error()})
		}

	// ── Agentic LLM response ───────────────────────────────────────────────
	case AgentLLMDoneMsg:
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
			cmds = append(cmds, agentRetrievalCmd(m.bridge, agentQuery, m.settings.FinalK, m.tuiState.Window, m.tuiState.Debug, m.tuiState.MinResults, m.tuiState.ScoreThreshold))

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
			})
			m.resetAgentState()

		default:
			// LLM didn't follow format — treat entire response as an answer.
			m.appState = StateIdle
			m.addMessage(ChatMessage{Role: "assistant", Content: msg.Payload})
			m.resetAgentState()
		}

	// ── Agentic retrieval result ───────────────────────────────────────────
	case AgentRetrievalDoneMsg:
		if msg.Err != nil {
			m.addMessage(ChatMessage{Role: "error", Content: "Agent retrieval error: " + msg.Err.Error()})
			m.appState = StateIdle
			m.resetAgentState()
			break
		}
		m.addMessage(ChatMessage{
			Role:    "agentic_step",
			Content: fmt.Sprintf("[retrieved %d chunks, top=%.3f]", msg.NumResults, msg.TopScore),
		})
		if m.tuiState.Debug {
			keyStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
			valStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite)
			dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
			var sb strings.Builder
			if msg.DebugStats != nil {
				ds := msg.DebugStats
				sb.WriteString(keyStyle.Render("  query_type") + dimStyle.Render(" = ") + valStyle.Render(ds.QueryType) + "\n")
				sb.WriteString(keyStyle.Render("  FAISS hits") + dimStyle.Render(" = ") + valStyle.Render(fmt.Sprintf("%d", ds.FaissHits)) +
					"   " + keyStyle.Render("BM25 hits") + dimStyle.Render(" = ") + valStyle.Render(fmt.Sprintf("%d", ds.BM25Hits)) + "\n")
				sb.WriteString(keyStyle.Render("  candidates") + dimStyle.Render(" = ") + valStyle.Render(fmt.Sprintf("%d", ds.TotalCandidates)) +
					"   " + keyStyle.Render("reranked") + dimStyle.Render(" → ") + valStyle.Render(fmt.Sprintf("top %d", ds.Reranked)) + "\n")
			}
			m.addMessage(ChatMessage{Role: "system", Content: "debug · agent retrieval stats\n" + sb.String()})
		}

		if msg.NumResults == 0 || msg.TopScore < 0.5 {
			m.agentFailed = append(m.agentFailed, msg.Query)
			m.agentConsecFail++
			m.agentCtx.WriteString(fmt.Sprintf("\n\n--- '%s' — low confidence, try different angle ---", msg.Query))
		} else {
			m.agentConsecFail = 0
			m.agentCtx.WriteString(fmt.Sprintf("\n\n--- Results for: '%s' ---\n%s", msg.Query, msg.ChunkText))
		}

		m.agentStep++

		if m.agentConsecFail >= 3 || m.agentStep >= m.agentHardCap() {
			m.addMessage(ChatMessage{Role: "agentic_step", Content: "[forcing final answer…]"})
			cmds = append(cmds, m.forceAgentAnswerCmd())
		} else {
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

	// ── Pipeline results ───────────────────────────────────────────────────
	case PipelineResultMsg:
		m.screen = ScreenChat
		if msg.Err != nil {
			m.addMessage(ChatMessage{Role: "error", Content: fmt.Sprintf("[pipeline/%s] %v", msg.Action, msg.Err)})
		} else {
			m.addMessage(ChatMessage{Role: "system", Content: fmt.Sprintf("[pipeline/%s]\n%s", msg.Action, msg.Message)})
		}

	// ── Chat file list ─────────────────────────────────────────────────────
	case ChatFileListMsg:
		if msg.Err != nil {
			m.addMessage(ChatMessage{Role: "error", Content: "Could not list raw files: " + msg.Err.Error()})
			m.screen = ScreenChat
		} else {
			m.chatFileList = msg.Files
			m.chatFileCursor = 0
		}

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
		// Global quit
		if msg.Type == tea.KeyCtrlC || msg.String() == "ctrl+q" {
			m.saveState()
			if m.bridge != nil {
				m.bridge.Close()
			}
			return m, tea.Quit
		}

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
		}
	}

	// Forward events to textarea when user can type (chat + help screens).
	if m.screen == ScreenChat || m.screen == ScreenHelp {
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
                }
        }
        return nil
}

func (m *AppModel) applyAPISettingsEdit() {
        switch m.apiSettingsEditing {
        case "api_key":
                if v := strings.TrimSpace(m.apiSettingsInput); v != "" {
                        m.settings.NIMAPIKey = v
                        _ = m.settings.Save()
                }
        case "base_url":
                if v := strings.TrimSpace(m.apiSettingsInput); v != "" {
                        m.settings.NIMBaseURL = v
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

func (m *AppModel) handlePipelineKey(msg tea.KeyMsg) []tea.Cmd {
        m.acConsumedKey = true
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
                        case "ingest":
                                return []tea.Cmd{pipelineIngestCmd(m.bridge, input)}
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
                case "test":
                        m.pipelineInputMode = true
                        m.pipelinePrompt = "Enter query"
                        m.pipelineInput = ""
                case "ingest":
                        m.pipelineInputMode = true
                        m.pipelinePrompt = "Enter file path"
                        m.pipelineInput = ""
                }
        }
        return nil
}

func (m AppModel) viewPipelineContent() string {
        var sb strings.Builder
        sb.WriteString(ui.TitleStyle.Render("Pipeline Management") + "\n\n")
        if m.pipelineInputMode {
                sb.WriteString(ui.SystemMsgStyle.Render(m.pipelinePrompt+": ") + ui.ActiveFlagStyle.Render(m.pipelineInput+"_"))
                sb.WriteString("\n\n")
                sb.WriteString(ui.HelpStyle.Render("Enter to confirm · ESC to cancel"))
        } else {
                for i, item := range pipelineMenuItems {
                        isSelected := i == m.pipelineCursor
                        marker := "  "
                        numStr := fmt.Sprintf("[%d]", i+1)
                        if isSelected {
                                marker = ui.ActiveFlagStyle.Render("▶ ")
                                numStr = ui.TitleStyle.Render(numStr)
                        } else {
                                numStr = ui.SystemMsgStyle.Render(numStr)
                        }
                        label := item.label
                        if isSelected {
                                label = lipgloss.NewStyle().Foreground(ui.ColorWhite).Render(label)
                        } else {
                                label = ui.SystemMsgStyle.Render(label)
                        }
                        sb.WriteString(fmt.Sprintf("%s%s %s\n", marker, numStr, label))
                }
                sb.WriteString("\n")
                sb.WriteString(ui.HelpStyle.Render("↑↓: navigate · Enter: run · ESC: back"))
        }
        return lipgloss.NewStyle().Height(m.viewportHeight()).Render(
                ui.BorderStyle.Width(m.width - 4).Render(sb.String()),
        )
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
	switch msg.Type {
	case tea.KeyEsc:
		m.screen = ScreenChat
		m.acConsumedKey = true
	case tea.KeyEnter:
		// Submit whatever is in the textarea and return to chat.
		m.acConsumedKey = true
		m.screen = ScreenChat
		return m.handleChatKey(msg)
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

	case "/pipeline":
		m.screen = ScreenPipeline
		m.pipelineCursor = 0
		m.pipelineInputMode = false
		m.pipelineInput = ""

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
		m.helpVp.Width = m.helpVpWidth()
		m.helpVp.Height = m.helpVpHeight()
		m.helpVp.SetContent(m.buildHelpContent())
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

	case "/clear":
		m.messages = nil
		m.renderMessages()

	default:
		m.addMessage(ChatMessage{Role: "system", Content: "unknown command. type /help for help."})
	}
	return nil
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
	cfg := m.settings.ToLLMConfig()

	return func() tea.Msg {
		system := buildAgentSystemPrompt(confident, maxSteps, len(searches))

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
			"%s\n\nQuestion: %s\n\nSearches done: %v\nLow-quality searches (avoid similar): %v\n%s\n\nIMPORTANT: Do NOT append words to previous queries. Think of a completely new angle.\nContext retrieved so far:\n%s\n\nWhat do you do next? (SEARCH: ... or ANSWER: ...)",
			system, question, searches, failed, remaining, ctxText,
		)

		messages := []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		}

		resp, err := workers.CallLLM(context.Background(), cfg, messages)
		if err != nil {
			return AgentLLMDoneMsg{Action: "ANSWER", Payload: fmt.Sprintf("[LLM error: %v]", err)}
		}

		action, payload := parseAgentResponse(resp)
		return AgentLLMDoneMsg{Action: action, Payload: payload}
	}
}

func (m *AppModel) forceAgentAnswerCmd() tea.Cmd {
	question := m.agentQuestion
	accumulated := m.agentCtx.String()
	numSearches := len(m.agentSearches)
	cfg := m.settings.ToLLMConfig()

	return func() tea.Msg {
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

		resp, err := workers.CallLLM(context.Background(), cfg, messages)
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
}

// ─── Streaming ────────────────────────────────────────────────────────────────

// startStreamingCmd wires up the streaming goroutine and returns a cmd to start
// collecting tokens.
func (m *AppModel) startStreamingCmd(retrievedContext string) []tea.Cmd {
	// RAG-only mode: if no API key is configured, skip LLM and show chunks directly.
	if m.settings.NIMAPIKey == "" || m.tuiState.RAGOnly {
		notice := "no API key set — showing retrieved chunks"
		if m.tuiState.RAGOnly {
			notice = "RAG-only mode — showing retrieved chunks"
		}
		m.addMessage(ChatMessage{Role: "system", Content: notice})

		// Format results as a readable table.
		keyStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan)
		valStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite)
		dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
		starStyle := lipgloss.NewStyle().Foreground(ui.ColorYellow).Bold(true)
		rankStyle := lipgloss.NewStyle().Foreground(ui.ColorGreen).Bold(true)

		var sb strings.Builder
		sb.WriteString(dimStyle.Render(fmt.Sprintf("  %-3s  %-7s  %-3s  %-8s  %-19s", "#", "score", "★", "src", "timestamp")) + "\n")
		sb.WriteString(dimStyle.Render("  " + strings.Repeat("─", 80)) + "\n")
		for _, r := range m.streamSources {
			score := r.RerankScore
			if score == 0 {
				score = r.Score
			}
			star := dimStyle.Render("·")
			if r.KeywordBoosted {
				star = starStyle.Render("★")
			}
			ts := r.Chunk.TimestampStart
			if len(ts) > 19 {
				ts = ts[:19]
			}
			src := r.Source
			if len(src) > 7 {
				src = src[:7]
			}
			text := r.Chunk.Text
			maxText := m.width - 50
			if maxText < 30 {
				maxText = 30
			}
			if len(text) > maxText {
				text = text[:maxText] + "…"
			}
			sb.WriteString(
				rankStyle.Render(fmt.Sprintf("  %2d", r.Rank)) +
					keyStyle.Render(fmt.Sprintf("  %7.4f  ", score)) +
					star +
					valStyle.Render(fmt.Sprintf("  %-8s %-19s", src, ts)) + "\n" +
					"      " + valStyle.Render(text) + "\n",
			)
		}
		m.appState = StateIdle
		m.addMessage(ChatMessage{
			Role:    "assistant",
			Content: sb.String(),
			Sources: m.streamSources,
		})
		m.streamSources = nil
		return nil
	}

	// Find the query from the last user message.
	query := ""
	for i := len(m.messages) - 1; i >= 0; i-- {
		if m.messages[i].Role == "user" {
			query = m.messages[i].Content
			break
		}
	}

	prompt := buildPrompt(query, retrievedContext, "")
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: prompt},
	}

	cfg := m.settings.ToLLMConfig()
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
	if len(m.messages) > 0 && m.messages[len(m.messages)-1].Role == "assistant" {
		m.messages[len(m.messages)-1].Content = m.streamBuf.String()
		m.messages[len(m.messages)-1].Streaming = true
	} else {
		m.messages = append(m.messages, ChatMessage{
			Role:      "assistant",
			Content:   m.streamBuf.String(),
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
		// Use warm amber colour while the message is still streaming.
		var header string
		if msg.Streaming {
			header = ui.AIStreamingMsgStyle.Render(label)
		} else {
			header = ui.AssistantMsgStyle.Render(label)
		}
		rendered := msg.Content
		if m.mdRenderer != nil && !msg.Streaming {
			if md, err := m.mdRenderer.Render(msg.Content); err == nil {
				rendered = strings.TrimRight(md, "\n")
			}
		}
		sb.WriteString(header + " " + rendered)
		if m.tuiState.ShowSources && len(msg.Sources) > 0 {
			sb.WriteString("\n")
			sb.WriteString(renderSourceTable(msg.Sources, m.tuiState.Window))
		}
		return sb.String()

	case "agentic_step":
		return ui.AgentStepStyle.Render("  " + msg.Content)

	case "system":
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
	case ScreenChats:
		content = m.viewChatsContent()
	case ScreenHelp:
		content = m.viewHelpContent()
	case ScreenModelPicker:
		content = m.viewModelPickerContent()
	case ScreenNick:
		content = m.viewNickContent()
	default: // ScreenChat
		if m.appState == StateStarting || len(m.messages) == 0 {
			content = m.viewSplash()
		} else {
			content = m.viewport.View()
		}
	}
	inputRow := m.renderInputRow()
	statusBar := m.renderStatusBar()
	return lipgloss.JoinVertical(lipgloss.Left, content, inputRow, statusBar)
}

func (m AppModel) renderStatusBar() string {
	chatName := m.activeChat
	if chatName == "" {
		chatName = "none"
	}
	modelShort := m.settings.NIMModel
	if idx := strings.LastIndex(modelShort, "/"); idx >= 0 {
		modelShort = modelShort[idx+1:]
	}

	// Build parts — always-visible items first.
	parts := []string{
		"chat=" + chatName,
		"model=" + modelShort,
		fmt.Sprintf("window=%d", m.tuiState.Window),
		fmt.Sprintf("k=%d", m.settings.FinalK),
	}

	// Flags shown only when active.
	if m.tuiState.Debug {
		parts = append(parts, ui.ActiveFlagStyle.Render("debug"))
	}
	if m.tuiState.ShowSources {
		parts = append(parts, ui.ActiveFlagStyle.Render("sources"))
	}
	if m.tuiState.AgentMode {
		parts = append(parts, ui.ActiveFlagStyle.Render("agent"))
	}
	if m.tuiState.Confident {
		parts = append(parts, ui.ActiveFlagStyle.Render("confident"))
	}
	if m.settings.ThinkingMode {
		parts = append(parts, ui.ActiveFlagStyle.Render("thinking"))
	}

	// Busy state indicator.
	switch m.appState {
	case StateStarting:
		parts = append(parts, ui.AgentStepStyle.Render("[starting…]"))
	case StateRetrieving:
		parts = append(parts, ui.AgentStepStyle.Render("[retrieving…]"))
	case StateStreaming:
		parts = append(parts, ui.AgentStepStyle.Render("[streaming…]"))
	case StateAgentStep:
		parts = append(parts, ui.AgentStepStyle.Render(fmt.Sprintf("[agent step %d]", m.agentStep+1)))
	case StateError:
		parts = append(parts, ui.ErrorMsgStyle.Render("[error]"))
	}

	bar := strings.Join(parts, "  ")
	return ui.StatusBarStyle.Width(m.width).Render(bar)
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
		parts = append(parts, ui.HelpStyle.Render("  ↑↓ PgUp/PgDn: scroll · ESC: close · type and press Enter to submit"))
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

// buildHelpContent returns the fully-styled text that is loaded into helpVp.
func (m AppModel) buildHelpContent() string {
	var modelList strings.Builder
	for i, mdl := range allModels {
		modelList.WriteString(fmt.Sprintf("  [%2d] %s\n", i+1, mdl))
	}

	titleStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
	sectionStyle := lipgloss.NewStyle().Foreground(ui.ColorGreen).Bold(true)
	cmdStyle := lipgloss.NewStyle().Foreground(ui.ColorYellow)
	dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)

	var sb strings.Builder
	sb.WriteString(titleStyle.Render("ragtag · Help") + "\n\n")

	sb.WriteString(sectionStyle.Render("Commands") + "\n")
	cmds := [][2]string{
		{"/debug", "toggle debug output"},
		{"/sources", "toggle source table in replies"},
		{"/confident", "toggle confident mode (agentic: more thorough)"},
		{"/window N", "set context window size"},
		{"/k N", "set final_k (chunks passed to LLM)"},
		{"/minresults N", "set minimum result count (adaptive threshold)"},
		{"/threshold F", "set score threshold (e.g. /threshold 3.5)"},
		{"/model N", "set model by number (e.g. /model 10 = deepseek-v3.2)"},
		{"/model <nick>", "set model by nickname (e.g. /model deepseek)"},
		{"/model", "open interactive model picker"},
		{"/agent", "toggle agentic mode"},
		{"/agent N q", "run agentic query with max N steps"},
		{"/thinking", "toggle thinking mode"},
		{"/rag", "toggle RAG-only mode (skip LLM)"},
		{"/pause", "pause / cancel the current AI operation"},
		{"/pipeline", "open pipeline management screen"},
		{"/view", "browse and read raw chat JSON files"},
		{"/chats", "switch between chats"},
		{"/settings", "open settings categories"},
		{"/stats", "show index statistics"},
		{"/help", "show this screen"},
	}
	for _, c := range cmds {
		sb.WriteString(fmt.Sprintf("  %-20s %s\n", cmdStyle.Render(c[0]), dimStyle.Render(c[1])))
	}

	sb.WriteString("\n" + sectionStyle.Render("Hotkeys") + "\n")
	hotkeys := [][2]string{
		{"Ctrl+C / Ctrl+Q", "quit"},
		{"ESC", "close overlay / clear input"},
		{"PgUp / PgDn", "scroll (also ↑↓ in pagers)"},
		{"Tab", "autocomplete command"},
		{"↑↓", "cycle autocomplete / navigate menus"},
	}
	for _, h := range hotkeys {
		sb.WriteString(fmt.Sprintf("  %-20s %s\n", cmdStyle.Render(h[0]), dimStyle.Render(h[1])))
	}

	sb.WriteString("\n" + sectionStyle.Render("Models") + "\n")
	sb.WriteString(dimStyle.Render(modelList.String()))

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
	return func() tea.Msg {
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
	return func() tea.Msg {
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

// ─── Pipeline cmd factories ───────────────────────────────────────────────────

// PipelineResultMsg carries the result of a pipeline operation.
type PipelineResultMsg struct {
	Action  string
	Message string
	Err     error
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

func pipelineIngestCmd(bridge *workers.Bridge, filePath string) tea.Cmd {
	return func() tea.Msg {
		if bridge == nil {
			return PipelineResultMsg{Action: "ingest", Err: fmt.Errorf("bridge not initialised")}
		}
		msg, err := bridge.Ingest(filePath)
		return PipelineResultMsg{Action: "ingest", Message: msg, Err: err}
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

func buildPrompt(query, retrievedContext, customTemplate string) string {
	if customTemplate != "" {
		result := strings.ReplaceAll(customTemplate, "{query}", query)
		result = strings.ReplaceAll(result, "{context}", retrievedContext)
		return result
	}
	return fmt.Sprintf(`You are an expert assistant. Answer the user's question using only the context provided below.

Question:
%s

Context chunks:
%s

Instructions:
- Provide a clear and concise answer.
- Use information only from the context.
- Include references to the chunks where relevant.
- If the answer is not in the context, say: "The information is not available in the provided context."
- Maintain clarity and correct pronoun references.

Answer:`, query, retrievedContext)
}

func buildAgentSystemPrompt(confident bool, maxSteps, searchesDone int) string {
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

	return fmt.Sprintf(`You are querying a semantic search index built from Discord chat logs between two people: 'peepee' (Devin) and 'sania'.
SEARCH queries use semantic similarity — short, natural phrases (3-6 words) work best.
Each SEARCH returns the most relevant chat message chunks from the logs.

Rules:
- Output  SEARCH: <query>    to retrieve chunks from the chat log
- Output  ANSWER: <response> when you have enough information
- %s
- Use short, conversational phrases as queries — NOT boolean expressions or long sentences
- Each SEARCH must be meaningfully different from previous ones
- Do NOT repeat the same query twice
- If a search returns low scores, it means the data doesn't contain that phrasing — pivot completely
- NEVER build on a previous query by appending words to it
- If you've done 3+ searches with no good results, just ANSWER with what you found
- After 3-4 searches with no useful results, give your best ANSWER based on what was found
- Do NOT keep searching if you already have enough context to answer`, budgetLine)
}

// parseAgentResponse scans the LLM response line-by-line for the first
// SEARCH: or ANSWER: directive and returns (action, payload).
func parseAgentResponse(response string) (action, payload string) {
	re := regexp.MustCompile(`(?i)^(SEARCH|ANSWER):\s*(.*)`)
	lines := strings.Split(response, "\n")
	var contentLines []string
	foundAction := ""

	for _, line := range lines {
		stripped := strings.TrimSpace(line)
		if foundAction == "" {
			if m := re.FindStringSubmatch(stripped); m != nil {
				foundAction = strings.ToUpper(m[1])
				if strings.TrimSpace(m[2]) != "" {
					contentLines = append(contentLines, strings.TrimSpace(m[2]))
				}
			}
		} else if foundAction == "ANSWER" {
			contentLines = append(contentLines, stripped)
		}
	}

	if foundAction == "" {
		return "ANSWER", strings.TrimSpace(response)
	}

	joined := strings.Join(contentLines, "\n")
	if foundAction == "SEARCH" && len(contentLines) > 0 {
		// Take only the first line, strip junk after quotes/newlines
		first := contentLines[0]
		parts := regexp.MustCompile(`[\n"']| to | in order to `).Split(first, 2)
		joined = strings.TrimSpace(parts[0])
	}

	return foundAction, joined
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
	h := m.height - 4 - acHeight
	if h < 5 {
		return 5
	}
	return h
}

// helpVpHeight returns the inner height of the help pager viewport.
// The outer border box uses viewportHeight() rows total; the rounded border
// consumes 2 (top + bottom), and we also account for the 1-line hint above
// the input box when the help screen is active.
func (m AppModel) helpVpHeight() int {
	// When the help screen is shown, renderInputRow appends one extra hint line.
	h := m.height - 4 - 1 - 2 // total - inputRow(3+statusBar(1)) - hintLine(1) - border(2)
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
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".json") {
				files = append(files, e.Name())
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
