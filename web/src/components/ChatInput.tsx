import { useState, KeyboardEvent } from "react";

interface Props {
  disabled?: boolean;
  onSend: (text: string) => void;
}

export function ChatInput({ disabled, onSend }: Props) {
  const [text, setText] = useState("");

  const submit = () => {
    const t = text.trim();
    if (!t || disabled) return;
    onSend(t);
    setText("");
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t border-border px-6 py-4 bg-background">
      <div className="flex gap-2 items-end">
        <textarea
          rows={2}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          disabled={disabled}
          placeholder="아이디어를 자연어로 적어주세요. Enter 전송, Shift+Enter 줄바꿈."
          className="flex-1 resize-none bg-background border border-input rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-1 disabled:opacity-50 placeholder:text-muted-foreground"
        />
        <button
          type="button"
          onClick={submit}
          disabled={disabled || !text.trim()}
          className="px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-md hover:bg-primary/90 disabled:opacity-40 transition-colors"
        >
          전송
        </button>
      </div>
    </div>
  );
}
