import { describe, expect, it, vi } from "vitest";
import { ApiError, createClient } from "../api/client";

describe("client", () => {
  it("throws ApiError with the server's error_code on a 404", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error_code: "ANALYSIS_NOT_FOUND", message: "No such analysis", detail: null }), {
        status: 404,
      }),
    );
    const client = createClient({ baseUrl: "http://api.test/", useMock: false, fetchImpl });
    const err = await client.getAnalysis("a_9").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.error_code).toBe("ANALYSIS_NOT_FOUND");
    expect(err.status).toBe(404);
    expect(fetchImpl).toHaveBeenCalledWith("http://api.test/api/v1/analysis/a_9", expect.objectContaining({ signal: expect.anything() }));
  });

  it("posts the question to /api/v1/analyze", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(new Response("{}", { status: 200 }));
    await createClient({ baseUrl: "http://api.test", useMock: false, fetchImpl }).analyze("Who?");
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe("http://api.test/api/v1/analyze");
    expect(JSON.parse(init.body)).toEqual({ question: "Who?" });
  });

  it("reports a network failure as NETWORK_ERROR", async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    const err = await createClient({ baseUrl: "http://api.test", useMock: false, fetchImpl })
      .listHistory()
      .catch((e) => e);
    expect(err.error_code).toBe("NETWORK_ERROR");
  });

  it("mock mode returns the sample analysis", async () => {
    const result = await createClient({ baseUrl: "", useMock: true }).analyze("anything");
    expect(result.claims.length).toBe(result.decisions.length);
  });
});
