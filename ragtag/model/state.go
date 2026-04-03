package model

import "github.com/emirate/rag-tui/workers"

// AppState represents the current operational state of the application.
type AppState int

const (
	StateIdle       AppState = iota // waiting for user input
	StateStarting                   // bridge is starting up
	StateRetrieving                 // waiting for Python retrieval
	StateStreaming                  // streaming LLM response
	StateAgentStep                  // agentic mode: one step running
	StateError                      // an error occurred
)

// Screen represents which overlay/view is currently active.
type Screen int

const (
	ScreenChat                Screen = iota // main chat view
	ScreenSettings                          // model settings editor
	ScreenRetrievalSettings                 // retrieval settings editor
	ScreenSettingsMenu                      // top-level settings menu (categories)
	ScreenChats                             // chat switcher overlay
	ScreenHelp                              // help overlay
	ScreenModelPicker                       // model picker overlay
	ScreenNick                              // nickname editor overlay
	ScreenAPISettings                       // API settings (key, url)
	ScreenInterfaceSettings                 // interface toggles
	ScreenPipeline                          // pipeline management
	ScreenChatList                          // raw chat file browser
	ScreenChatViewer                        // paged chat message viewer
)

// ChatMessage is one entry in the visible chat log.
type ChatMessage struct {
	Role          string           // "user", "assistant", "system", "agentic_step", "error"
	Content       string
	Sources       []workers.Result // populated for assistant messages when show_sources=true
	Searches      []string         // for agentic answers
	NumSearches   int
	Streaming     bool // true while content is actively being streamed
}
