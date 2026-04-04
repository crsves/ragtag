package model

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/emirate/rag-tui/ui"
)

// ─── Types ────────────────────────────────────────────────────────────────────

const (
	ClarifyKindLLM     = "llm"     // re-runs the pipeline with clarification context
	ClarifyKindRestart = "restart" // restarts the binary after /update
	ClarifyKindUpdate  = "update"  // startup update notification prompt
)

type ClarifyOption struct {
	ID    string `json:"id"`
	Label string `json:"label"`
}

// ClarifyState holds everything needed to render and drive the clarify overlay.
type ClarifyState struct {
	Active           bool
	Kind             string // ClarifyKindLLM | ClarifyKindRestart
	Question         string
	Options          []ClarifyOption
	Cursor           int
	AllowFreeInput   bool
	SuggestedDefault string
	FreeInputMode    bool
	FreeInputText    string
	OriginalQuery    string // kept so we can re-run with clarification
}

// PanelHeight returns the number of terminal rows the clarify panel occupies.
func (cs *ClarifyState) PanelHeight() int {
	// 2 border + 1 question + 1 blank + N options + 1 blank + 1 hint
	h := 2 + 1 + 1 + len(cs.Options) + 1 + 1
	if cs.AllowFreeInput {
		h += 2 // blank line + input/prompt line
	}
	return h
}

// ─── Parser ───────────────────────────────────────────────────────────────────

// ParseClarifyOutput detects {"mode":"clarify",...} in the LLM response.
func ParseClarifyOutput(content string) (*ClarifyState, bool) {
	trimmed := strings.TrimSpace(content)
	if !strings.HasPrefix(trimmed, "{") {
		return nil, false
	}
	var out struct {
		Mode             string          `json:"mode"`
		Question         string          `json:"question"`
		Options          []ClarifyOption `json:"options"`
		AllowFreeInput   bool            `json:"allow_free_input"`
		SuggestedDefault string          `json:"suggested_default"`
	}
	if err := json.Unmarshal([]byte(trimmed), &out); err != nil {
		return nil, false
	}
	if out.Mode != "clarify" || len(out.Options) == 0 {
		return nil, false
	}
	cursor := 0
	for i, opt := range out.Options {
		if opt.ID == out.SuggestedDefault {
			cursor = i
			break
		}
	}
	return &ClarifyState{
		Active:           true,
		Kind:             ClarifyKindLLM,
		Question:         out.Question,
		Options:          out.Options,
		Cursor:           cursor,
		AllowFreeInput:   out.AllowFreeInput,
		SuggestedDefault: out.SuggestedDefault,
	}, true
}

// ─── Renderer ─────────────────────────────────────────────────────────────────

// RenderClarify builds the clarify overlay panel.
func RenderClarify(cs *ClarifyState, width int) string {
	cardWidth := width - 4
	if cardWidth < 48 {
		cardWidth = 48
	}

	accentStyle  := lipgloss.NewStyle().Foreground(ui.ColorYellow).Bold(true)
	dimStyle     := lipgloss.NewStyle().Foreground(ui.ColorDim)
	cursorStyle  := lipgloss.NewStyle().Foreground(ui.ColorYellow).Bold(true)
	selectedStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
	unselStyle   := lipgloss.NewStyle().Foreground(ui.ColorWhite)
	idDimStyle   := lipgloss.NewStyle().Foreground(ui.ColorDim)
	idSelStyle   := lipgloss.NewStyle().Foreground(ui.ColorCyan)
	inputStyle   := lipgloss.NewStyle().Foreground(ui.ColorWhite)

	var rows []string

	// Question line
	rows = append(rows, accentStyle.Render("? ")+unselStyle.Render(cs.Question))
	rows = append(rows, "")

	for i, opt := range cs.Options {
		if i == cs.Cursor {
			rows = append(rows, fmt.Sprintf("  %s  %s  %s",
				cursorStyle.Render("❯"),
				idSelStyle.Render("["+opt.ID+"]"),
				selectedStyle.Render(opt.Label),
			))
		} else {
			rows = append(rows, fmt.Sprintf("  %s  %s  %s",
				" ",
				idDimStyle.Render("["+opt.ID+"]"),
				unselStyle.Render(opt.Label),
			))
		}
	}

	if cs.AllowFreeInput {
		rows = append(rows, "")
		if cs.FreeInputMode {
			rows = append(rows, inputStyle.Render("  ▸ "+cs.FreeInputText+"█"))
		} else {
			rows = append(rows, dimStyle.Render("  or type a custom answer…"))
		}
	}

	rows = append(rows, "")
	rows = append(rows, dimStyle.Render("  ↑↓ navigate  ·  Enter select  ·  Esc dismiss"))

	body := strings.Join(rows, "\n")
	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(ui.ColorYellow).
		Padding(0, 1).
		Width(cardWidth).
		Render(body)
}

// ─── Key handler ──────────────────────────────────────────────────────────────

// handleClarifyKey processes key events when the clarify overlay is active.
func (m *AppModel) handleClarifyKey(msg tea.KeyMsg) []tea.Cmd {
	var cmds []tea.Cmd

	if m.clarify.FreeInputMode {
		switch msg.Type {
		case tea.KeyEnter:
			// Submit free-form answer
			text := strings.TrimSpace(m.clarify.FreeInputText)
			if text == "" {
				m.clarify.FreeInputMode = false
				return cmds
			}
			m.clarify.Active = false
			m.clarify.FreeInputMode = false
			cmds = append(cmds, m.submitClarifyAnswer("custom", text)...)
		case tea.KeyEsc:
			m.clarify.FreeInputMode = false
		case tea.KeyBackspace:
			if len(m.clarify.FreeInputText) > 0 {
				runes := []rune(m.clarify.FreeInputText)
				m.clarify.FreeInputText = string(runes[:len(runes)-1])
			}
		default:
			if msg.Type == tea.KeyRunes || msg.Type == tea.KeySpace {
				m.clarify.FreeInputText += msg.String()
			}
		}
		m.viewport.Height = m.viewportHeight()
		m.renderMessages()
		return cmds
	}

	switch msg.Type {
	case tea.KeyUp:
		if m.clarify.Cursor > 0 {
			m.clarify.Cursor--
		}
	case tea.KeyDown:
		if m.clarify.Cursor < len(m.clarify.Options)-1 {
			m.clarify.Cursor++
		}
	case tea.KeyEnter:
		opt := m.clarify.Options[m.clarify.Cursor]
		m.clarify.Active = false
		cmds = append(cmds, m.submitClarifyAnswer(opt.ID, opt.Label)...)
	case tea.KeyEsc:
		m.clarify.Active = false
	default:
		// Typing activates free-input mode (when allowed).
		if m.clarify.AllowFreeInput && (msg.Type == tea.KeyRunes || msg.Type == tea.KeySpace) {
			m.clarify.FreeInputMode = true
			m.clarify.FreeInputText = msg.String()
		}
	}

	m.viewport.Height = m.viewportHeight()
	m.renderMessages()
	return cmds
}

// submitClarifyAnswer dispatches the selected option.
func (m *AppModel) submitClarifyAnswer(id, label string) []tea.Cmd {
	var cmds []tea.Cmd

	switch m.clarify.Kind {
	case ClarifyKindRestart:
		if id == "yes" {
			m.shouldRestart = true
			m.saveState()
			if m.bridge != nil {
				m.bridge.Close()
			}
			cmds = append(cmds, tea.Quit)
		}
		// "no" → just dismiss (already done above)

	case ClarifyKindUpdate:
		if id == "yes" {
			m.addMessage(ChatMessage{Role: "system", Content: "Updating ragtag…"})
			cmds = append(cmds, selfUpdateCmd())
		}

	case ClarifyKindLLM:
		// Re-run retrieval+LLM with original query appended with the clarification.
		orig := m.clarify.OriginalQuery
		refined := orig + "\n[Clarification: " + label + "]"
		// Show a brief user-side indicator of the choice.
		m.addMessage(ChatMessage{
			Role:    "user",
			Content: "→ " + label,
		})
		// Kick off retrieval (which will then stream the LLM response).
		if m.cancelFn != nil {
			m.cancelFn()
		}
		m.activeCtx, m.cancelFn = context.WithCancel(context.Background())
		m.appState = StateRetrieving
		cmds = append(cmds,
			m.spinner.Tick,
			retrieveCmd(m.bridge, refined, m.settings.FinalK, m.tuiState.Window, m.tuiState.Debug, m.tuiState.MinResults, m.tuiState.ScoreThreshold),
		)
	}

	return cmds
}

// lastUserQuery returns the most recent user message content.
func (m *AppModel) lastUserQuery() string {
	for i := len(m.messages) - 1; i >= 0; i-- {
		if m.messages[i].Role == "user" {
			return m.messages[i].Content
		}
	}
	return ""
}

// ShouldRestart reports whether the app should exec a fresh copy of itself.
func (m AppModel) ShouldRestart() bool {
	return m.shouldRestart
}
