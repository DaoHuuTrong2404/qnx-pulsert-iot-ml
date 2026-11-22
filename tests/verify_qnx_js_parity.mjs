import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import test from "node:test";
import vm from "node:vm";

const htmlPath = resolve(import.meta.dirname, "..", "docs", "index.html");
const html = readFileSync(htmlPath, "utf8");

function extractBalancedBlock(source, anchor, openCh, closeCh) {
	const start = source.indexOf(anchor);
	assert.notEqual(start, -1, `Missing anchor: ${anchor}`);
	const firstBrace = source.indexOf(openCh, start);
	let depth = 0;
	for (let i = firstBrace; i < source.length; i++) {
		if (source[i] === openCh) depth++;
		else if (source[i] === closeCh) {
			depth--;
			if (depth === 0) return source.slice(start, i + 1);
		}
	}
	throw new Error(`Unclosed block for ${anchor}`);
}

const w1Block = extractBalancedBlock(html, "const W1 =", "[", "]");
const b1Block = extractBalancedBlock(html, "const B1 =", "[", "]");
const w2Block = extractBalancedBlock(html, "const W2 =", "[", "]");
const b2Line = html.match(/const B2 = [^;]+;/)?.[0];
assert.ok(b2Line, "Missing B2 constant");

const q15Fn = extractBalancedBlock(html, "function runQ15Mlp(", "{", "}");
const rtaFn = extractBalancedBlock(html, "function computeRta(", "{", "}");
const simFn = extractBalancedBlock(
	html,
	"function simulateFifoWindow(",
	"{",
	"}",
);

const sandbox = {};
vm.runInNewContext(
	`${w1Block};\n${b1Block};\n${w2Block};\n${b2Line}\n${q15Fn}\n${rtaFn}\n${simFn}\nglobalThis.__exports = { runQ15Mlp, computeRta, simulateFifoWindow };`,
	sandbox,
);

const { runQ15Mlp, computeRta, simulateFifoWindow } = sandbox.__exports;

test("QNX docs/index.html exact RTA and 1-us FIFO sim match [265, 645, 1550]", () => {
	const nominal = [
		{
			name: "sensor_ingest_fifo",
			priority: 40,
			wcet: 220,
			period: 2000,
			deadline: 2000,
			blocking: 45,
		},
		{
			name: "qnx_ml_infer_srv",
			priority: 30,
			wcet: 380,
			period: 2000,
			deadline: 2000,
			blocking: 45,
		},
		{
			name: "aps_telemetry_log",
			priority: 12,
			wcet: 950,
			period: 10000,
			deadline: 10000,
			blocking: 0,
		},
	];
	const rta = computeRta(nominal);
	const sim = simulateFifoWindow(nominal, 10000, 45);
	assert.equal(rta.exactSched, true);
	assert.deepEqual(
		rta.tasks.map((t) => t.response),
		[265, 645, 1550],
	);
	assert.deepEqual(Array.from(sim.maxResp), [265, 645, 1550]);
	assert.deepEqual(Array.from(sim.misses), [0, 0, 0]);
	assert.equal(rta.util.toFixed(4), "0.3950");
	assert.equal(rta.llBound.toFixed(4), "0.7798");
});

test("QNX docs/index.html detects priority-inversion overload deadline miss", () => {
	const overloaded = [
		{
			name: "sensor_ingest_fifo",
			priority: 40,
			wcet: 820,
			period: 2000,
			deadline: 2000,
			blocking: 240,
		},
		{
			name: "qnx_ml_infer_srv",
			priority: 30,
			wcet: 1050,
			period: 2000,
			deadline: 2000,
			blocking: 240,
		},
		{
			name: "aps_telemetry_log",
			priority: 12,
			wcet: 950,
			period: 10000,
			deadline: 10000,
			blocking: 0,
		},
	];
	const rta = computeRta(overloaded);
	const sim = simulateFifoWindow(overloaded, 10000, 240);
	assert.equal(rta.exactSched, false);
	assert.equal(sim.misses.reduce((a, b) => a + b, 0) > 0, true);
});

test("QNX docs/index.html Q15 fixed-point MLP classifies NOMINAL (<0.54) and CRITICAL_TRIP (>=0.75)", () => {
	const nominalScore = runQ15Mlp([4600, 5200, 4100, 3900]) / 32768;
	const warningScore = runQ15Mlp([8000, 8500, 8500, 8000]) / 32768;
	const faultScore = runQ15Mlp([26500, 18200, 14500, 27400]) / 32768;
	assert.equal(nominalScore < 0.54, true, `nominal=${nominalScore}`);
	assert.equal(
		warningScore >= 0.54 && warningScore < 0.75,
		true,
		`warning=${warningScore}`,
	);
	assert.equal(faultScore >= 0.75, true, `fault=${faultScore}`);
});
