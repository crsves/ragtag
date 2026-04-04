package model

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/emirate/rag-tui/workers"
)

// Settings holds all NIM/retrieval configuration parsed from nim_config.py.
type Settings struct {
	NIMAPIKey       string
	NIMBaseURL      string
	HFToken         string
	NIMModel        string
	Temperature     float64
	TopP            float64
	MaxTokens       int
	FinalK          int
	MaxContextChars int
	ThinkingMode    bool

	path string // path to nim_config.py
}

var defaults = Settings{
	NIMAPIKey:       "",
	NIMBaseURL:      "https://integrate.api.nvidia.com/v1",
	HFToken:         "",
	NIMModel:        "meta/llama-3.3-70b-instruct",
	Temperature:     0.2,
	TopP:            0.7,
	MaxTokens:       1024,
	FinalK:          10,
	MaxContextChars: 300000,
	ThinkingMode:    false,
}

// LoadSettings reads nim_config.py from the given path and parses all settings.
// Missing values fall back to defaults.
func LoadSettings(nimConfigPath string) (*Settings, error) {
	s := defaults
	s.path = nimConfigPath

	data, err := os.ReadFile(nimConfigPath)
	if err != nil {
		// Return defaults if the file doesn't exist.
		if os.IsNotExist(err) {
			return &s, nil
		}
		return nil, fmt.Errorf("read nim_config: %w", err)
	}

	text := string(data)

	// String fields
	if v := parseString(text, "NIM_API_KEY"); v != "" {
		s.NIMAPIKey = v
	}
	if v := parseString(text, "NIM_BASE_URL"); v != "" {
		s.NIMBaseURL = v
	}
	if v := parseString(text, "HF_TOKEN"); v != "" {
		s.HFToken = v
	}
	if v := parseString(text, "NIM_MODEL"); v != "" {
		s.NIMModel = v
	}

	// Float fields
	if v, ok := parseFloat(text, "TEMPERATURE"); ok {
		s.Temperature = v
	}
	if v, ok := parseFloat(text, "TOP_P"); ok {
		s.TopP = v
	}

	// Int fields
	if v, ok := parseInt(text, "MAX_TOKENS"); ok {
		s.MaxTokens = v
	}
	if v, ok := parseInt(text, "FINAL_K"); ok {
		s.FinalK = v
	}
	if v, ok := parseInt(text, "MAX_CONTEXT_CHARS"); ok {
		s.MaxContextChars = v
	}

	// Boolean
	re := regexp.MustCompile(`(?m)^THINKING_MODE\s*=\s*(True|False)`)
	if m := re.FindStringSubmatch(text); m != nil {
		s.ThinkingMode = m[1] == "True"
	}

	return &s, nil
}

// Save writes the current settings back to nim_config.py.
func (s *Settings) Save() error {
	thinking := "False"
	if s.ThinkingMode {
		thinking = "True"
	}

	text := strings.Join([]string{
		"# NIM API configuration — gitignored, do not commit",
		fmt.Sprintf(`NIM_API_KEY = "%s"`, s.NIMAPIKey),
		fmt.Sprintf(`NIM_BASE_URL = "%s"`, s.NIMBaseURL),
		fmt.Sprintf(`HF_TOKEN = "%s"`, s.HFToken),
		fmt.Sprintf(`NIM_MODEL = "%s"`, s.NIMModel),
		"",
		"# Generation settings — low temperature keeps RAG answers factual",
		fmt.Sprintf("TEMPERATURE = %g", s.Temperature),
		fmt.Sprintf("TOP_P = %g", s.TopP),
		fmt.Sprintf("MAX_TOKENS = %d", s.MaxTokens),
		"",
		"# Retrieval settings",
		fmt.Sprintf("FINAL_K = %d        # chunks passed to LLM after reranking", s.FinalK),
		"# 128K token context window → rough safety limit before we warn",
		fmt.Sprintf("MAX_CONTEXT_CHARS = %d   # ~75K tokens at 4 chars/token", s.MaxContextChars),
		"# Thinking mode (DeepSeek V3.2 only)",
		fmt.Sprintf("THINKING_MODE = %s", thinking),
		"",
	}, "\n")

	return os.WriteFile(s.path, []byte(text), 0644)
}

// ToLLMConfig converts settings into a workers.LLMConfig.
func (s *Settings) ToLLMConfig() workers.LLMConfig {
	return workers.LLMConfig{
		APIKey:       s.NIMAPIKey,
		BaseURL:      s.NIMBaseURL,
		Model:        s.NIMModel,
		Temperature:  float32(s.Temperature),
		TopP:         float32(s.TopP),
		MaxTokens:    s.MaxTokens,
		ThinkingMode: s.ThinkingMode,
	}
}

// ─── helpers ─────────────────────────────────────────────────────────────────

func parseString(text, key string) string {
	re := regexp.MustCompile(fmt.Sprintf(`(?m)^%s\s*=\s*["'](.+?)["']`, regexp.QuoteMeta(key)))
	if m := re.FindStringSubmatch(text); m != nil {
		return m[1]
	}
	return ""
}

func parseFloat(text, key string) (float64, bool) {
	re := regexp.MustCompile(fmt.Sprintf(`(?m)^%s\s*=\s*([0-9.]+)`, regexp.QuoteMeta(key)))
	if m := re.FindStringSubmatch(text); m != nil {
		v, err := strconv.ParseFloat(m[1], 64)
		if err == nil {
			return v, true
		}
	}
	return 0, false
}

func parseInt(text, key string) (int, bool) {
	re := regexp.MustCompile(fmt.Sprintf(`(?m)^%s\s*=\s*([0-9_]+)`, regexp.QuoteMeta(key)))
	if m := re.FindStringSubmatch(text); m != nil {
		clean := strings.ReplaceAll(m[1], "_", "")
		v, err := strconv.Atoi(clean)
		if err == nil {
			return v, true
		}
	}
	return 0, false
}

// ─── TUI State ───────────────────────────────────────────────────────────────

// TUIState holds ephemeral UI preferences persisted to .tui_state.json.
type TUIState struct {
	Debug            bool    `json:"debug"`
	ShowSources      bool    `json:"show_sources"`
	Window           int     `json:"window"`
	Confident        bool    `json:"confident"`
	AgentMode        bool    `json:"agent_mode"`
	AgentToolSearch  bool    `json:"agent_tool_search"`
	AgentToolContext bool    `json:"agent_tool_context"`
	RAGOnly          bool    `json:"rag_only"`
	MinResults       int     `json:"min_results"`     // adaptive retrieval floor
	ScoreThreshold   float64 `json:"score_threshold"` // rerank score cutoff (0 = disabled)
	OutputMode       string  `json:"output_mode"`     // "plain" | "structured" | "rich"
}

// LoadTUIState reads the JSON state file. Returns sensible defaults on any error.
func LoadTUIState(path string) TUIState {
	s := TUIState{
		Window:           5,
		AgentToolSearch:  true,
		AgentToolContext: true,
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return s
	}
	_ = json.Unmarshal(data, &s)
	if s.Window == 0 {
		s.Window = 5
	}
	return s
}

// SaveTUIState writes the TUI state to the given JSON file.
func SaveTUIState(path string, s TUIState) error {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}
