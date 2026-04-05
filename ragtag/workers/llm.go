package workers

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
)

// LLMConfig holds configuration for the NIM/OpenAI-compatible LLM client.
type LLMConfig struct {
	APIKey       string
	BaseURL      string
	Model        string
	Temperature  float32
	TopP         float32
	MaxTokens    int
	ThinkingMode bool
	LogDir       string
}

// newClient creates an OpenAI client configured for the given LLMConfig.
func newClient(cfg LLMConfig) *openai.Client {
	config := openai.DefaultConfig(cfg.APIKey)
	config.BaseURL = cfg.BaseURL
	return openai.NewClientWithConfig(config)
}

// extraBody returns the extra_body map for thinking-mode requests, or nil.
func extraBody(cfg LLMConfig) map[string]interface{} {
	if cfg.ThinkingMode && strings.Contains(cfg.Model, "deepseek") {
		return map[string]interface{}{
			"chat_template_kwargs": map[string]interface{}{
				"thinking": true,
			},
		}
	}
	return nil
}

// StreamLLM streams completion tokens to tokenCh.
// When streaming ends (successfully or with an error) it closes tokenCh and
// sends to errCh (nil on success). The caller must drain tokenCh.
func StreamLLM(ctx context.Context, cfg LLMConfig, messages []openai.ChatCompletionMessage, tokenCh chan<- string, errCh chan<- error) {
	client := newClient(cfg)
	Logf(cfg.LogDir, "stream llm start model=%s messages=%d summary=%s", cfg.Model, len(messages), summarizeMessages(messages))

	req := openai.ChatCompletionRequest{
		Model:       cfg.Model,
		Messages:    messages,
		Temperature: cfg.Temperature,
		TopP:        cfg.TopP,
		MaxTokens:   cfg.MaxTokens,
		Stream:      true,
	}

	if eb := extraBody(cfg); eb != nil {
		// go-openai exposes extra fields via the request struct in recent versions.
		// We encode them into the request body via StreamOptions or a custom approach.
		// Since go-openai doesn't have a generic ExtraBody field, we skip this for
		// streaming (thinking mode is primarily useful for non-streaming calls).
		_ = eb
	}

	stream, err := client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		Logf(cfg.LogDir, "stream llm create error: %v", err)
		close(tokenCh)
		errCh <- fmt.Errorf("create stream: %w", err)
		return
	}
	defer stream.Close()

	for {
		resp, err := stream.Recv()
		if err != nil {
			close(tokenCh)
			if err.Error() == "EOF" || strings.Contains(err.Error(), "EOF") {
				Logf(cfg.LogDir, "stream llm done: eof")
				errCh <- nil
			} else {
				Logf(cfg.LogDir, "stream llm recv error: %v", err)
				errCh <- fmt.Errorf("stream recv: %w", err)
			}
			return
		}
		if len(resp.Choices) > 0 {
			tok := resp.Choices[0].Delta.Content
			if tok != "" {
				select {
				case tokenCh <- tok:
				case <-ctx.Done():
					close(tokenCh)
					errCh <- ctx.Err()
					return
				}
			}
		}
	}
}

// CallLLM performs a non-streaming chat completion and returns the full response.
// It retries up to 2 times on empty-choices responses (transient API errors).
func CallLLM(ctx context.Context, cfg LLMConfig, messages []openai.ChatCompletionMessage) (string, error) {
	client := newClient(cfg)
	Logf(cfg.LogDir, "call llm start model=%s messages=%d summary=%s", cfg.Model, len(messages), summarizeMessages(messages))

	req := openai.ChatCompletionRequest{
		Model:       cfg.Model,
		Messages:    messages,
		Temperature: cfg.Temperature,
		TopP:        cfg.TopP,
		MaxTokens:   cfg.MaxTokens,
		Stream:      false,
	}

	const maxAttempts = 3
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		resp, err := client.CreateChatCompletion(ctx, req)
		if err != nil {
			Logf(cfg.LogDir, "call llm error attempt=%d: %v", attempt, err)
			if attempt == maxAttempts {
				return "", fmt.Errorf("chat completion: %w", err)
			}
			time.Sleep(time.Duration(attempt) * 500 * time.Millisecond)
			continue
		}

		if len(resp.Choices) == 0 {
			Logf(cfg.LogDir, "call llm empty choices attempt=%d id=%s usage=%+v raw=%+v", attempt, resp.ID, resp.Usage, resp)
			if attempt == maxAttempts {
				return "", fmt.Errorf("no choices in response")
			}
			time.Sleep(time.Duration(attempt) * 500 * time.Millisecond)
			continue
		}

		content := strings.TrimSpace(resp.Choices[0].Message.Content)
		Logf(cfg.LogDir, "call llm success attempt=%d id=%s choices=%d finish=%s content=%q", attempt, resp.ID, len(resp.Choices), resp.Choices[0].FinishReason, clipLog(content, 400))
		return content, nil
	}
	return "", fmt.Errorf("no choices in response after %d attempts", maxAttempts)
}
