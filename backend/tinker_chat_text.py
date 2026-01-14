import asyncio
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from queue import Queue, Empty

from agent.agent import BellaAgent
from agent.state import SessionState


class TextChatApp:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.root = tk.Tk()
        self.root.title("Bella Concierge – Text Chat")
        self.caller_name_var = tk.StringVar(value="Guest")
        self.input_var = tk.StringVar()
        self.agent: BellaAgent | None = None
        self.message_queue = Queue()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.poll_queue()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="Caller Name").grid(row=0, column=0, sticky="w")
        name_entry = ttk.Entry(frame, textvariable=self.caller_name_var, width=30)
        name_entry.grid(row=0, column=1, sticky="we")
        name_entry.bind("<Return>", self.start_session)

        start_button = ttk.Button(
            frame, text="Start Session", command=self.start_session
        )
        start_button.grid(row=0, column=2, padx=(8, 0))

        self.log_box = scrolledtext.ScrolledText(frame, height=20, state="disabled")
        self.log_box.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=8)

        input_row = ttk.Frame(frame)
        input_row.grid(row=2, column=0, columnspan=3, sticky="we")

        self.input_entry = ttk.Entry(input_row, textvariable=self.input_var, width=60)
        self.input_entry.grid(row=0, column=0, sticky="we")
        self.input_entry.bind("<Return>", self.send_message)
        self.input_entry.configure(state="disabled")

        self.send_button = ttk.Button(
            input_row, text="Send", command=self.send_message
        )
        self.send_button.grid(row=0, column=1, padx=(8, 0))
        self.send_button.configure(state="disabled")

        frame.columnconfigure(1, weight=1)
        input_row.columnconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def start_session(self, _=None) -> None:
        name = self.caller_name_var.get().strip()
        if len(name) < 2:
            messagebox.showerror("Invalid name", "Enter at least two characters.")
            return

        state = SessionState(caller_name=name)
        self.agent = BellaAgent(state)
        self._append("assistant", f"Hi {name.title()}, I'm Maria. How may I help you?")
        self.input_entry.configure(state="normal")
        self.send_button.configure(state="normal")

    def send_message(self, _=None) -> None:
        if not self.agent:
            messagebox.showinfo("Start session", "Please start a session first.")
            return
        text = self.input_var.get().strip()
        if not text:
            return
        self.input_var.set("")
        self._append("user", text)
        
        # Run the async agent call in the separate thread's event loop
        self.loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._respond(text))
        )

    async def _respond(self, text: str) -> None:
        """Coroutine to handle the agent's response."""
        try:
            reply = await self.agent.respond(text)
            self.message_queue.put(("assistant", reply))
        except Exception as e:
            self.message_queue.put(("error", f"An error occurred: {e}"))

    def poll_queue(self):
        """Check the queue for messages from the async thread."""
        try:
            while True:
                speaker, text = self.message_queue.get_nowait()
                if speaker == "error":
                    messagebox.showerror("Agent Error", text)
                else:
                    self._append(speaker, text)
        except Empty:
            pass
        self.root.after(100, self.poll_queue)


    def _append(self, speaker: str, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert(tk.END, f"{speaker.title()}: {text}\n\n")
        self.log_box.configure(state="disabled")
        self.log_box.see(tk.END)
    
    def on_closing(self):
        """Handle window closing."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_async_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Runs the asyncio event loop."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


if __name__ == "__main__":
    # Set up a new asyncio event loop that will run in a separate thread
    async_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=run_async_loop, args=(async_loop,), daemon=True)
    thread.start()
    
    # The Tkinter app runs in the main thread
    app = TextChatApp(loop=async_loop)
    app.run()
