/*
 * TrustLedger-XAI - E5 cross-implementation determinism, step 2 of 2 (Node.js).
 *     npm install canonicalize
 *     node crossimpl_verify.mjs real|stress
 * Appends one row per serialisation to results/E5_crossimpl.csv.
 */
import { createHash } from 'node:crypto';
import { readFileSync, appendFileSync, existsSync, mkdirSync } from 'node:fs';
import canonicalize from 'canonicalize';

if (!String.prototype.isWellFormed) {
  Object.defineProperty(String.prototype, 'isWellFormed', {
    value: function () { return !/\p{Surrogate}/u.test(String(this)); }, writable: true, configurable: true });
}
const mode = process.argv[2] || 'real';
const sha = (s) => createHash('sha256').update(s).digest('hex');
const sortKeys = (o) => Array.isArray(o) ? o.map(sortKeys)
  : (o && typeof o === 'object') ? Object.keys(o).sort().reduce((a, k) => (a[k] = sortKeys(o[k]), a), {}) : o;

const arts = readFileSync(`crossimpl_${mode}.ndjson`, 'utf8').trim().split(/\r?\n/).map((l) => JSON.parse(l));
const py = readFileSync(`digests_python_${mode}.csv`, 'utf8').trim().split(/\r?\n/).slice(1)
  .map((l) => { const [i, c, n, k] = l.split(','); return { c, n, k: (k || '').trim() }; });

let ca = 0, na = 0, predicted = 0, predictedAndFailed = 0;
arts.forEach((a, i) => {
  if (sha(canonicalize(a)) === py[i].c) ca++;
  const ok = sha(JSON.stringify(sortKeys(a))) === py[i].n;
  if (ok) na++;
  if (py[i].k) { predicted++; if (!ok) predictedAndFailed++; }
});
const n = arts.length;
if (!existsSync('results')) mkdirSync('results');
const f = 'results/E5_crossimpl.csv';
if (!existsSync(f)) appendFileSync(f, 'set,node_version,serialisation,n,agree,agree_pct,artifacts_with_divergent_floats,of_which_disagreed\n');
appendFileSync(f, `${mode},${process.version},JCS/RFC8785 (rfc8785 vs canonicalize),${n},${ca},${(100*ca/n).toFixed(1)},,\n`);
appendFileSync(f, `${mode},${process.version},sorted-key JSON,${n},${na},${(100*na/n).toFixed(1)},${predicted},${predictedAndFailed}\n`);
console.log({ mode, n, canonicalAgree: ca, naiveAgree: na, predicted, predictedAndFailed });
