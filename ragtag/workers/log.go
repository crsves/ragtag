package workers

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
)

func Logf(logDir, format string, args ...interface{}) {
	path := logPath(logDir)
	if path == "" {
		return
	}
	_ = os.MkdirAll(filepath.Dir(path), 0755)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return
	}
	defer f.Close()
	msg := fmt.Sprintf(format, args...)
	fmt.Fprintf(f, "[%s] %s\n", time.Now().Format(time.RFC3339), msg)
}

func logPath(logDir string) string {
	if strings.TrimSpace(logDir) != "" {
		return filepath.Join(logDir, "ragtag.log")
	}
	if v := os.Getenv("RAG_DIR"); strings.TrimSpace(v) != "" {
		return filepath.Join(v, "ragtag.log")
	}
	if cwd, err := os.Getwd(); err == nil {
		return filepath.Join(cwd, "ragtag.log")
	}
	return ""
}

func clipLog(s string, max int) string {
	s = strings.Join(strings.Fields(s), " ")
	if max <= 0 || len([]rune(s)) <= max {
		return s
	}
	r := []rune(s)
	return string(r[:max]) + "…"
}

func summarizeMessages(messages []openai.ChatCompletionMessage) string {
	parts := make([]string, 0, len(messages))
	for _, msg := range messages {
		parts = append(parts, fmt.Sprintf("%s=%q", msg.Role, clipLog(msg.Content, 160)))
	}
	return strings.Join(parts, " | ")
}
