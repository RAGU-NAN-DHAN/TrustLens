import { describe, expect, it } from "vitest";
import { limitLevel } from "../components/LimitNotice";

describe("limitLevel", () => {
  it("is none without notes", () => {
    expect(limitLevel(undefined)).toBe("none");
    expect(limitLevel([])).toBe("none");
  });
  it("is switched when a backup key took over", () => {
    expect(limitLevel(["Gemini key 1 hit its limit (HTTP 429); switched to backup key 2."])).toBe("switched");
  });
  it("is degraded when a verifier or search was unavailable", () => {
    expect(limitLevel(["Gemini key 1 hit its limit (HTTP 429); switched to backup key 2.", "Gemini verifier was unavailable (HTTP 429)"])).toBe("degraded");
  });
});
