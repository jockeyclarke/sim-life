import { type ControlMessage, type ServerMessage, type Snapshot, decodeSnapshot, parseServerMessage } from "./protocol";

export type ConnectionState = "connected" | "disconnected";

export interface Handlers {
  onSnapshot(snapshot: Snapshot): void;
  onMessage(message: ServerMessage): void;
  onState(state: ConnectionState): void;
}

const RETRY_MS = 2000;

/** A WebSocket to the engine that keeps trying to reconnect. */
export class Connection {
  private socket: WebSocket | null = null;

  constructor(
    private readonly url: string,
    private readonly handlers: Handlers,
  ) {
    this.open();
  }

  send(message: ControlMessage): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    }
  }

  private open(): void {
    const socket = new WebSocket(this.url);
    socket.binaryType = "arraybuffer";
    this.socket = socket;

    socket.onopen = () => this.handlers.onState("connected");
    socket.onclose = () => {
      this.handlers.onState("disconnected");
      setTimeout(() => this.open(), RETRY_MS);
    };
    socket.onmessage = (event: MessageEvent) => {
      if (event.data instanceof ArrayBuffer) {
        try {
          this.handlers.onSnapshot(decodeSnapshot(event.data));
        } catch (error) {
          // keep showing the last good frame
          console.warn("dropped a snapshot that could not be decoded:", error);
        }
      } else if (typeof event.data === "string") {
        const message = parseServerMessage(event.data);
        if (message) this.handlers.onMessage(message);
        else console.warn("ignored an unrecognised message from the engine");
      }
    };
  }
}
