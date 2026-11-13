import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

/** A fixture file's bytes as a standalone ArrayBuffer. Test-only. */
export function readFixture(name: string): ArrayBuffer {
  const path = fileURLToPath(new URL(`./__fixtures__/${name}`, import.meta.url));
  const bytes = readFileSync(path);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}
