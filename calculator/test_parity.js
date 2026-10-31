// Compare predict.js + models.json against scikit-learn probabilities (see test_parity.py).
// Note: models.json was fitted on median-imputed data while the parity run refits on
// median-filled data, so both use identical training matrices and settings.
const fs = require("fs");
const { predictAll } = require("./predict.js");
const bundle = JSON.parse(fs.readFileSync("models.json", "utf8"));
const cases = JSON.parse(fs.readFileSync("parity_cases.json", "utf8"));
const expected = JSON.parse(fs.readFileSync("parity_expected.json", "utf8"));

let worst = {}, bad = 0, total = 0;
for (const key of Object.keys(expected)) {
  const [outcome, set] = key.split("|");
  cases.forEach((c, i) => {
    const got = predictAll(bundle, outcome, set === "phq", c);
    for (const m of Object.keys(got)) {
      const diff = Math.abs(got[m] - expected[key][m][i]);
      worst[m] = Math.max(worst[m] || 0, diff);
      total++;
      if (diff > 1e-3) bad++;
    }
  });
}
console.log("comparisons:", total, " over 0.001 apart:", bad);
console.log("largest absolute difference by model:", JSON.stringify(worst));
process.exit(bad === 0 ? 0 : 1);
