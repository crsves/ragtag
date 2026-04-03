package ui

import "github.com/charmbracelet/lipgloss"

// Terminal colour palette — pastel/earthen tones.
var (
	ColorCyan     = lipgloss.Color("#b8d8ba") // sage green  — primary accent
	ColorGreen    = lipgloss.Color("#d9dbbc") // yellow-green — active / success
	ColorYellow   = lipgloss.Color("#fcddbc") // peach        — highlights / state
	ColorRed      = lipgloss.Color("#ef959d") // rose         — errors / warnings
	ColorDim      = lipgloss.Color("#69585f") // dark mauve   — muted / secondary text
	ColorWhite    = lipgloss.Color("255")     // terminal white
	ColorMagenta  = lipgloss.Color("#ef959d") // same as red
	ColorAIActive = lipgloss.Color("#f0a070") // warm amber   — AI is working
)

// Pre-built lipgloss styles used throughout the TUI.
var (
	StatusBarStyle = lipgloss.NewStyle().
			Background(lipgloss.Color("#69585f")).
			Foreground(ColorWhite).
			Padding(0, 1)

	UserMsgStyle = lipgloss.NewStyle().
			Foreground(ColorCyan).
			Bold(true)

	AssistantMsgStyle = lipgloss.NewStyle().
				Foreground(ColorWhite)

	AIStreamingMsgStyle = lipgloss.NewStyle().
				Foreground(ColorAIActive)

	SystemMsgStyle = lipgloss.NewStyle().
			Foreground(ColorDim).
			Italic(true)

	ErrorMsgStyle = lipgloss.NewStyle().
			Foreground(ColorRed)

	AgentStepStyle = lipgloss.NewStyle().
			Foreground(ColorYellow)

	SourceTableStyle = lipgloss.NewStyle().
				Foreground(ColorDim).
				Border(lipgloss.RoundedBorder()).
				BorderForeground(ColorDim).
				Padding(0, 1)

	HelpStyle = lipgloss.NewStyle().
			Foreground(ColorDim)

	ActiveFlagStyle = lipgloss.NewStyle().
			Foreground(ColorGreen).
			Bold(true)

	InactiveFlagStyle = lipgloss.NewStyle().
				Foreground(ColorDim)

	TitleStyle = lipgloss.NewStyle().
			Foreground(ColorCyan).
			Bold(true)

	BorderStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(ColorCyan).
			Padding(0, 1)

	AIActiveBorderStyle = lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(ColorAIActive).
				Padding(0, 1)

	ContextAnchorStyle = lipgloss.NewStyle().
				Foreground(ColorYellow).
				Bold(true)
)
