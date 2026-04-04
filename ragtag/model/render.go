package model

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/emirate/rag-tui/ui"
)

// ─── Output schemas injected into the LLM prompt ─────────────────────────────

const richOutputSchema = `
OUTPUT FORMAT — you MUST respond with ONLY a valid JSON object. No text outside the JSON.

{
  "mode": "rich",
  "components": [
    {
      "type": "summary",
      "title": "Answer",
      "content": "Your main answer in plain prose."
    },
    {
      "type": "chat_log",
      "title": "#channel-name or 'Conversation'",
      "messages": [
        {"user": "username", "timestamp": "HH:MM", "content": "message text"}
      ]
    },
    {
      "type": "table",
      "title": "Table title",
      "columns": ["Column A", "Column B"],
      "rows": [["value", "value"]]
    },
    {
      "type": "code_block",
      "language": "log",
      "content": "raw content here"
    },
    {
      "type": "key_value",
      "title": "Details",
      "data": {"Key": "value", "Key2": "value2"}
    },
    {
      "type": "list",
      "title": "Items",
      "items": ["item one", "item two"],
      "ordered": false
    },
    {
      "type": "timeline",
      "title": "What happened",
      "events": [{"time": "10:21", "content": "event description"}]
    }
  ]
}

RULES — strictly follow:
- NEVER use markdown tables (|---|---|). Use the "table" component instead.
- NEVER use markdown headers (## or ###). Use component "title" fields.
- NEVER use markdown bold (**text**). Just write the text.
- Always include at least one "summary" component with your answer.
- Use "chat_log" whenever you quote actual messages from the context.
- Use "table" only for comparative/structured data — not for chat messages.
- Extract real timestamps and usernames from the context chunks when quoting.
- Keep "summary" content concise — the other components carry the detail.
`

const structuredOutputSchema = `
OUTPUT FORMAT — respond with ONLY a valid JSON object:

{
  "mode": "structured",
  "summary": "One-paragraph answer in plain prose.",
  "key_points": ["concise point", "another point"]
}

RULES:
- NEVER use markdown tables (|---|---|).
- NEVER use markdown headers.
- Keep summary under 3 sentences.
- key_points should be 2-5 short bullet items.
`

const clarifySchema = `
CLARIFICATION — if (and ONLY if) the query is genuinely ambiguous with multiple distinct plausible intents, you MAY respond with ONLY this JSON instead of an answer:

{
  "mode": "clarify",
  "question": "Which do you mean?",
  "options": [
    {"id": "A", "label": "Specific option A"},
    {"id": "B", "label": "Specific option B"}
  ],
  "suggested_default": "A",
  "allow_free_input": false
}

STRICT rules for clarify:
- Only trigger for GENUINE ambiguity (2+ distinct intents that produce very different answers).
- NEVER clarify obvious queries, follow-ups, or when context makes intent clear.
- Provide 2–4 mutually exclusive, specific options. Never vague options.
- Do NOT ask "Can you clarify?" in plain text — use the JSON format or answer directly.
- If in doubt, make a reasonable assumption and answer.
`

const plainOutputRules = `
RULES for your answer:
- NEVER use markdown tables (|---|---|). Write data as prose or simple lists instead.
- NEVER use markdown headers (## or ###).
- Write in clear, plain prose.
`

// ─── Parser ───────────────────────────────────────────────────────────────────

// ParseRichOutput attempts to parse the LLM response as structured JSON output.
// Returns (rendered string, true) on success, ("", false) if plain text.
func ParseRichOutput(content string, width int) (string, bool) {
	trimmed := strings.TrimSpace(content)
	if !strings.HasPrefix(trimmed, "{") {
		return "", false
	}
	var out map[string]interface{}
	if err := json.Unmarshal([]byte(trimmed), &out); err != nil {
		return "", false
	}
	mode, _ := out["mode"].(string)
	switch mode {
	case "rich":
		return renderRich(out, width), true
	case "structured":
		return renderStructured(out, width), true
	}
	return "", false
}

// ─── Rich renderer ────────────────────────────────────────────────────────────

func renderRich(out map[string]interface{}, width int) string {
	components, _ := out["components"].([]interface{})
	var parts []string
	for _, c := range components {
		comp, ok := c.(map[string]interface{})
		if !ok {
			continue
		}
		if r := renderComponent(comp, width); r != "" {
			parts = append(parts, r)
		}
	}
	return strings.Join(parts, "\n")
}

func renderStructured(out map[string]interface{}, width int) string {
	summary, _ := out["summary"].(string)
	keyPoints, _ := out["key_points"].([]interface{})

	cardWidth := width - 4
	if cardWidth < 40 {
		cardWidth = 40
	}

	var parts []string
	if summary != "" {
		parts = append(parts, renderSummaryCard("Answer", summary, cardWidth))
	}
	if len(keyPoints) > 0 {
		items := make([]string, 0, len(keyPoints))
		for _, kp := range keyPoints {
			if s, ok := kp.(string); ok {
				items = append(items, s)
			}
		}
		parts = append(parts, renderListCard("Key Points", items, false, cardWidth))
	}
	return strings.Join(parts, "\n")
}

// ─── Component dispatcher ─────────────────────────────────────────────────────

func renderComponent(comp map[string]interface{}, width int) string {
	cardWidth := width - 4
	if cardWidth < 40 {
		cardWidth = 40
	}
	typ, _ := comp["type"].(string)
	switch typ {
	case "summary":
		title, _ := comp["title"].(string)
		content, _ := comp["content"].(string)
		return renderSummaryCard(title, content, cardWidth)
	case "chat_log":
		title, _ := comp["title"].(string)
		msgs, _ := comp["messages"].([]interface{})
		return renderChatLogCard(title, msgs, cardWidth)
	case "table":
		title, _ := comp["title"].(string)
		cols, _ := comp["columns"].([]interface{})
		rows, _ := comp["rows"].([]interface{})
		return renderTableCard(title, cols, rows, cardWidth)
	case "code_block":
		lang, _ := comp["language"].(string)
		content, _ := comp["content"].(string)
		return renderCodeBlockCard(lang, content, cardWidth)
	case "key_value":
		title, _ := comp["title"].(string)
		data, _ := comp["data"].(map[string]interface{})
		return renderKeyValueCard(title, data, cardWidth)
	case "list":
		title, _ := comp["title"].(string)
		items, _ := comp["items"].([]interface{})
		ordered, _ := comp["ordered"].(bool)
		strs := make([]string, 0, len(items))
		for _, it := range items {
			if s, ok := it.(string); ok {
				strs = append(strs, s)
			}
		}
		return renderListCard(title, strs, ordered, cardWidth)
	case "timeline":
		title, _ := comp["title"].(string)
		events, _ := comp["events"].([]interface{})
		return renderTimelineCard(title, events, cardWidth)
	}
	return ""
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

func wrapText(text string, width int) []string {
	if width <= 0 {
		width = 40
	}
	runes := []rune(strings.Join(strings.Fields(text), " "))
	var lines []string
	for len(runes) > width {
		cut := width
		for cut > 0 && runes[cut] != ' ' {
			cut--
		}
		if cut == 0 {
			cut = width
		}
		lines = append(lines, string(runes[:cut]))
		runes = []rune(strings.TrimLeft(string(runes[cut:]), " "))
	}
	if len(runes) > 0 {
		lines = append(lines, string(runes))
	}
	return lines
}

func cardBorder(borderColor lipgloss.Color, width int) lipgloss.Style {
	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(borderColor).
		Padding(0, 1).
		Width(width)
}

func sectionTitle(title string) string {
	return lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true).Render(title)
}

func dimSep(width int) string {
	inner := width - 6 // border(2) + padding(4)
	if inner < 1 {
		inner = 1
	}
	return lipgloss.NewStyle().Foreground(ui.ColorDim).Render(strings.Repeat("─", inner))
}

// ─── Summary ──────────────────────────────────────────────────────────────────

func renderSummaryCard(title, content string, cardWidth int) string {
	inner := cardWidth - 6
	textStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite)

	lines := wrapText(content, inner)
	body := sectionTitle(title) + "\n" +
		dimSep(cardWidth) + "\n" +
		textStyle.Render(strings.Join(lines, "\n"))

	return cardBorder(ui.ColorCyan, cardWidth).Render(body)
}

// ─── Chat log ─────────────────────────────────────────────────────────────────

func renderChatLogCard(title string, messages []interface{}, cardWidth int) string {
	inner := cardWidth - 6
	userStyle := lipgloss.NewStyle().Foreground(ui.ColorGreen).Bold(true)
	tsStyle   := lipgloss.NewStyle().Foreground(ui.ColorDim)
	textStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite)

	var rows []string
	for _, m := range messages {
		msg, ok := m.(map[string]interface{})
		if !ok {
			continue
		}
		user, _ := msg["user"].(string)
		ts, _   := msg["timestamp"].(string)
		text, _ := msg["content"].(string)

		prefix := userStyle.Render(user) + "  " + tsStyle.Render(ts) + "  "
		prefixLen := len([]rune(user)) + len([]rune(ts)) + 4
		textWidth := inner - prefixLen
		if textWidth < 20 {
			textWidth = 20
		}
		wrapped := wrapText(text, textWidth)
		for i, line := range wrapped {
			if i == 0 {
				rows = append(rows, prefix+textStyle.Render(line))
			} else {
				rows = append(rows, strings.Repeat(" ", prefixLen)+textStyle.Render(line))
			}
		}
	}

	body := sectionTitle(title) + "\n" +
		dimSep(cardWidth) + "\n" +
		strings.Join(rows, "\n")

	return cardBorder(ui.ColorDim, cardWidth).Render(body)
}

// ─── Table ────────────────────────────────────────────────────────────────────

func renderTableCard(title string, columns []interface{}, rows []interface{}, cardWidth int) string {
	inner := cardWidth - 6
	headerStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
	cellStyle   := lipgloss.NewStyle().Foreground(ui.ColorWhite)
	dimStyle    := lipgloss.NewStyle().Foreground(ui.ColorDim)

	cols := make([]string, len(columns))
	for i, c := range columns {
		cols[i], _ = c.(string)
	}

	// compute column widths
	colW := make([]int, len(cols))
	for i, c := range cols {
		colW[i] = len([]rune(c))
	}
	for _, row := range rows {
		r, ok := row.([]interface{})
		if !ok {
			continue
		}
		for i, cell := range r {
			if i >= len(colW) {
				break
			}
			s := fmt.Sprintf("%v", cell)
			if l := len([]rune(s)); l > colW[i] {
				colW[i] = l
			}
		}
	}
	// cap total to inner width
	totalW := 0
	for _, w := range colW {
		totalW += w + 2
	}
	if totalW > inner {
		scale := float64(inner) / float64(totalW)
		for i := range colW {
			colW[i] = int(float64(colW[i]) * scale)
			if colW[i] < 4 {
				colW[i] = 4
			}
		}
	}

	formatRow := func(cells []string, style lipgloss.Style) string {
		var sb strings.Builder
		for i, cell := range cells {
			w := 0
			if i < len(colW) {
				w = colW[i]
			}
			runes := []rune(cell)
			if len(runes) > w {
				runes = append(runes[:w-1], '…')
			}
			padded := string(runes) + strings.Repeat(" ", w-len(runes)+2)
			sb.WriteString(style.Render(padded))
		}
		return sb.String()
	}

	var tableRows []string
	tableRows = append(tableRows, formatRow(cols, headerStyle))
	sep := dimStyle.Render(strings.Repeat("─", inner))
	tableRows = append(tableRows, sep)
	for _, row := range rows {
		r, ok := row.([]interface{})
		if !ok {
			continue
		}
		cells := make([]string, len(r))
		for i, c := range r {
			cells[i] = fmt.Sprintf("%v", c)
		}
		tableRows = append(tableRows, formatRow(cells, cellStyle))
	}

	body := sectionTitle(title) + "\n" +
		dimSep(cardWidth) + "\n" +
		strings.Join(tableRows, "\n")

	return cardBorder(ui.ColorGreen, cardWidth).Render(body)
}

// ─── Code block ───────────────────────────────────────────────────────────────

func renderCodeBlockCard(language, content string, cardWidth int) string {
	inner := cardWidth - 6
	codeStyle := lipgloss.NewStyle().Foreground(ui.ColorAIActive)
	dimStyle  := lipgloss.NewStyle().Foreground(ui.ColorDim)

	label := "code"
	if language != "" {
		label = language
	}

	lines := strings.Split(content, "\n")
	var codeLines []string
	for _, line := range lines {
		runes := []rune(line)
		if len(runes) > inner {
			runes = append(runes[:inner-1], '…')
		}
		codeLines = append(codeLines, codeStyle.Render(string(runes)))
	}

	title := dimStyle.Render("❮ ") + lipgloss.NewStyle().Foreground(ui.ColorYellow).Bold(true).Render(label) + dimStyle.Render(" ❯")
	body := title + "\n" +
		dimSep(cardWidth) + "\n" +
		strings.Join(codeLines, "\n")

	return lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(ui.ColorDim).
		Padding(0, 1).
		Width(cardWidth).
		Render(body)
}

// ─── Key-value ────────────────────────────────────────────────────────────────

func renderKeyValueCard(title string, data map[string]interface{}, cardWidth int) string {
	inner := cardWidth - 6
	keyStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan)
	valStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite)
	dimStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)

	// find longest key for alignment
	maxKey := 0
	for k := range data {
		if l := len([]rune(k)); l > maxKey {
			maxKey = l
		}
	}
	if maxKey > inner/2 {
		maxKey = inner / 2
	}

	var rows []string
	for k, v := range data {
		keyPad := k + strings.Repeat(" ", maxKey-len([]rune(k)))
		valStr := fmt.Sprintf("%v", v)
		rows = append(rows, keyStyle.Render(keyPad)+"  "+dimStyle.Render("·")+"  "+valStyle.Render(valStr))
	}

	body := sectionTitle(title) + "\n" +
		dimSep(cardWidth) + "\n" +
		strings.Join(rows, "\n")

	return cardBorder(ui.ColorYellow, cardWidth).Render(body)
}

// ─── List ─────────────────────────────────────────────────────────────────────

func renderListCard(title string, items []string, ordered bool, cardWidth int) string {
	inner := cardWidth - 6
	bulletStyle := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
	textStyle   := lipgloss.NewStyle().Foreground(ui.ColorWhite)

	var rows []string
	for i, item := range items {
		bullet := bulletStyle.Render("·")
		if ordered {
			bullet = bulletStyle.Render(fmt.Sprintf("%d.", i+1))
		}
		prefixLen := 3
		textWidth := inner - prefixLen
		if textWidth < 10 {
			textWidth = 10
		}
		wrapped := wrapText(item, textWidth)
		for j, line := range wrapped {
			if j == 0 {
				rows = append(rows, bullet+" "+textStyle.Render(line))
			} else {
				rows = append(rows, strings.Repeat(" ", prefixLen)+textStyle.Render(line))
			}
		}
	}

	body := sectionTitle(title) + "\n" +
		dimSep(cardWidth) + "\n" +
		strings.Join(rows, "\n")

	return cardBorder(ui.ColorDim, cardWidth).Render(body)
}

// ─── Timeline ─────────────────────────────────────────────────────────────────

func renderTimelineCard(title string, events []interface{}, cardWidth int) string {
	inner := cardWidth - 6
	dotStyle  := lipgloss.NewStyle().Foreground(ui.ColorCyan).Bold(true)
	pipeStyle := lipgloss.NewStyle().Foreground(ui.ColorDim)
	tsStyle   := lipgloss.NewStyle().Foreground(ui.ColorYellow)
	textStyle := lipgloss.NewStyle().Foreground(ui.ColorWhite)

	var rows []string
	for i, e := range events {
		ev, ok := e.(map[string]interface{})
		if !ok {
			continue
		}
		t, _       := ev["time"].(string)
		content, _ := ev["content"].(string)

		prefix := dotStyle.Render("◆") + " " + tsStyle.Render(t) + "  "
		prefixLen := 2 + len([]rune(t)) + 2
		textWidth := inner - prefixLen
		if textWidth < 10 {
			textWidth = 10
		}
		wrapped := wrapText(content, textWidth)
		for j, line := range wrapped {
			if j == 0 {
				rows = append(rows, prefix+textStyle.Render(line))
			} else {
				rows = append(rows, pipeStyle.Render("│")+" "+strings.Repeat(" ", prefixLen-2)+textStyle.Render(line))
			}
		}
		if i < len(events)-1 {
			rows = append(rows, pipeStyle.Render("│"))
		}
	}

	body := sectionTitle(title) + "\n" +
		dimSep(cardWidth) + "\n" +
		strings.Join(rows, "\n")

	return cardBorder(ui.ColorDim, cardWidth).Render(body)
}

// ─── Streaming placeholder ────────────────────────────────────────────────────

// IsStreamingJSON returns true if the partial streaming content looks like
// the start of a rich JSON output — so we can show a placeholder instead of raw JSON.
func IsStreamingJSON(partial string) bool {
	t := strings.TrimSpace(partial)
	return strings.HasPrefix(t, "{") && strings.Contains(t, `"mode"`)
}
