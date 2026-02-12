import { useState } from "react";
import "./App.css";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // const sendMessage = async () => {
  //   if (!input.trim() || loading) return;

  //   const userPrompt = input;

  //   setMessages((prev) => [
  //     ...prev,
  //     { role: "user", content: userPrompt },
  //     {
  //       role: "assistant", content: (
  //         <div className="flex items-center gap-2">
  //           <span className="font-bold">Thinking</span>
  //           <div className="loader" />
  //         </div>
  //       )
  //     }
  //   ]);

  //   setInput("");
  //   setLoading(true);

  //   try {
  //     const res = await fetch("http://localhost:8000/generate", {
  //       method: "POST",
  //       headers: { "Content-Type": "application/json" },
  //       body: JSON.stringify({ prompt: userPrompt })
  //     });

  //     const data = await res.json();
  //     console.log(data);

  //     setMessages((prev) => [
  //       ...prev.slice(0, -1), // remove "Thinking..."
  //       {
  //         role: "assistant",
  //         content: data.response
  //       },
  //       {
  //         role: "assistant",
  //         files: data.files
  //       }
  //     ]);
  //   } catch (err) {
  //     setMessages((prev) => [
  //       ...prev.slice(0, -1),
  //       { role: "assistant", content: "❌ Something went wrong." }
  //     ]);
  //   } finally {
  //     setLoading(false);
  //   }
  // };
  const sendMessage = async () => {
  if (!input.trim() || loading) return;

  const userPrompt = input;

  setMessages((prev) => [
    ...prev,
    { role: "user", content: userPrompt },
    { role: "assistant", content: (
          <div className="flex items-center gap-2">
            <div className="loader" />
          </div>
        ) }
  ]);

  setInput("");
  setLoading(true);

  try {
    const res = await fetch("http://localhost:8000/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: userPrompt })
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let fullMessage = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split("\n\n").filter(Boolean);

      for (let line of lines) {
        if (!line.startsWith("data: ")) continue;

        const data = JSON.parse(line.replace("data: ", ""));

        switch (data.event) {

          case "intent":
            console.log("Intent:", data.value);
            break;

          case "chunk":
          case "msg":
            fullMessage += data.content;

            setMessages((prev) => [
              ...prev.slice(0, -1),
              { role: "assistant", content: fullMessage }
            ]);
            break;

          case "progress":
            setMessages((prev) => [
              ...prev.slice(0, -1),
              { role: "assistant", content: data.message }
            ]);
            break;

          case "file_written":
            console.log("Created:", data.file);
            break;

          case "app":
            setMessages((prev) => [
              ...prev,
              {
                role: "assistant",
                files: data.files
              }
            ]);
            break;

          case "done":
            setLoading(false);
            break;

          default:
            break;
        }
      }
    }
  } catch (err) {
    setMessages((prev) => [
      ...prev.slice(0, -1),
      { role: "assistant", content: "❌ Something went wrong." }
    ]);
    setLoading(false);
  }
};


  return (
    <div className="h-screen flex bg-gray-100">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r hidden md:flex flex-col">
        <div className="p-4 font-bold text-xl">AI BOT</div>
        <button className="m-4 p-2 rounded bg-black text-white">
          + New Chat
        </button>
        <div className="flex-1 overflow-auto px-4 text-sm text-gray-500">
          No conversations yet
        </div>
      </aside>

      {/* Chat Area */}
      <main className="flex-1 flex flex-col w-[calc(100vw-256px)]">
        {/* Header */}
        <header className="h-14 bg-white border-b flex items-center px-4 font-bold justify-center">
          Helper AI
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((msg, i) => (
            <div key={i}>
              {msg.content && (
                <div
                  className={`max-w-xl px-4 py-3 rounded-2xl text-sm shadow-sm ${
                    msg.role === "user"
                      ? "ml-auto bg-black text-white"
                      : "bg-white text-gray-800"
                  }`}
                >
                  {msg.content}
                </div>
              )}

              {/* Render generated files */}
              {msg.files && (
                <div className="bg-white border rounded-xl p-4 max-w-3xl space-y-4 text-left">
                  <h3 className="font-semibold text-sm text-gray-700">
                    Generated Files
                  </h3>

                  {Object.entries(msg.files).map(([filename, code]) => (
                    <div key={filename}>
                      <div className="text-xs font-mono text-gray-600 mb-1">
                        {filename}
                      </div>
                      <pre className="bg-gray-900 text-gray-100 text-xs p-3 rounded-lg overflow-x-auto">
                        <code>{code}</code>
                      </pre>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Input */}
        <div className="border-t bg-white p-4">
          <div className="flex items-center gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Describe your app..."
              className="flex-1 border rounded-xl px-4 py-2 focus:outline-none"
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            />
            <button
              onClick={sendMessage}
              disabled={loading}
              className="p-2 rounded-xl bg-black text-white disabled:opacity-50"
            >
              send
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
