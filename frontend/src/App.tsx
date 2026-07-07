import { ChangeEvent, FormEvent, useMemo, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000';

type Preview = {
  filename: string;
  rows: number;
  columns: string[];
  preview: Record<string, unknown>[];
};

type FeatureScore = {
  feature: string;
  rank: number;
  selected: boolean;
  stat_score: number;
  rf_importance: number;
  semantic_relevance: number;
  leakage_risk: number;
  availability: number;
  uncertainty: number;
  lasfs_score: number;
  reason: string;
};

type MethodResult = {
  method: string;
  selected_features: string[];
  rankings: FeatureScore[];
};

type Analysis = {
  filename: string;
  target: string;
  rows: number;
  feature_count: number;
  selected_count: number;
  target_classes: string[];
  scoring_mode: string;
  methods: MethodResult[];
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [target, setTarget] = useState('');
  const [kRatio, setKRatio] = useState(0.4);
  const [deepseekApiKey, setDeepseekApiKey] = useState('');
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const canAnalyze = Boolean(file && target && preview);

  const comparisonRows = useMemo(() => {
    if (!analysis) return [];
    const rf = analysis.methods.find((method) => method.method === 'Random Forest');
    const lasfs = analysis.methods.find((method) => method.method === 'LASFS');
    return analysis.selected_count > 0
      ? Array.from({ length: analysis.selected_count }, (_, index) => ({
          rank: index + 1,
          rf: rf?.selected_features[index] ?? '',
          lasfs: lasfs?.selected_features[index] ?? '',
        }))
      : [];
  }, [analysis]);

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setPreview(null);
    setAnalysis(null);
    setError('');
    setTarget('');
    if (!selected) return;
    await loadPreview(selected);
  }

  async function loadPreview(selected: File) {
    setLoading(true);
    try {
      const form = new FormData();
      form.append('file', selected);
      const response = await fetch(`${API_BASE}/api/preview`, {
        method: 'POST',
        body: form,
      });
      const payload = await parseJson(response);
      setPreview(payload as Preview);
      const columns = (payload as Preview).columns;
      setTarget(columns[columns.length - 1] ?? '');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not preview the CSV.');
    } finally {
      setLoading(false);
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!file || !target) return;
    setLoading(true);
    setError('');
    setAnalysis(null);
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('target', target);
      form.append('k_ratio', String(kRatio));
      if (deepseekApiKey.trim()) {
        form.append('deepseek_api_key', deepseekApiKey.trim());
      }
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: form,
      });
      setAnalysis((await parseJson(response)) as Analysis);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <div className="intro">
          <p className="eyebrow">LASFS Feature Selection</p>
          <h1>Upload a CSV and compare Random Forest with leakage-aware selection.</h1>
        </div>

        <form className="control-panel" onSubmit={handleSubmit}>
          <label className="upload-box">
            <span>{file ? file.name : 'Choose CSV file'}</span>
            <input type="file" accept=".csv,text/csv" onChange={handleFileChange} />
          </label>

          <label>
            Target column
            <select value={target} onChange={(event) => setTarget(event.target.value)} disabled={!preview}>
              <option value="">Select target</option>
              {preview?.columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </label>

          <label>
            Selected feature ratio: {Math.round(kRatio * 100)}%
            <input
              type="range"
              min="0.05"
              max="1"
              step="0.05"
              value={kRatio}
              onChange={(event) => setKRatio(Number(event.target.value))}
            />
          </label>

          <label>
            DeepSeek API Key
            <input
              className="api-key-input"
              type="password"
              placeholder="Optional, offline mode when empty"
              value={deepseekApiKey}
              onChange={(event) => setDeepseekApiKey(event.target.value)}
              autoComplete="off"
            />
          </label>

          <button type="submit" disabled={!canAnalyze || loading}>
            {loading ? 'Running...' : 'Run Selection'}
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        {preview && (
          <section className="summary-grid">
            <Metric label="Rows" value={preview.rows.toLocaleString()} />
            <Metric label="Columns" value={preview.columns.length.toLocaleString()} />
            <Metric label="Default Target" value={target || 'None'} />
            <Metric label="Scoring" value={deepseekApiKey.trim() ? 'DeepSeek' : 'Offline'} />
          </section>
        )}

        {analysis && (
          <>
            <section className="result-header">
              <div>
                <p className="eyebrow">Analysis Result</p>
                <h2>{analysis.filename}</h2>
              </div>
              <div className="target-pill">
                {analysis.target} - {analysis.target_classes.join(' / ')} - {analysis.scoring_mode}
              </div>
            </section>

            <section className="comparison">
              <div className="table-wrap">
                <h3>Selected Features</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Random Forest</th>
                      <th>LASFS</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparisonRows.map((row) => (
                      <tr key={row.rank}>
                        <td>{row.rank}</td>
                        <td>{row.rf}</td>
                        <td>{row.lasfs}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="method-grid">
              {analysis.methods.map((method) => (
                <MethodPanel key={method.method} method={method} />
              ))}
            </section>
          </>
        )}
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MethodPanel({ method }: { method: MethodResult }) {
  return (
    <section className="method-panel">
      <h3>{method.method} Ranking</h3>
      <div className="ranking-list">
        {method.rankings.slice(0, 12).map((feature) => (
          <article className={featureRowClass(feature)} key={feature.feature}>
            <div className="feature-title">
              <strong>
                {feature.rank}. {feature.feature}
              </strong>
              {feature.selected && <span>selected</span>}
            </div>
            <div className="score-line">
              <Score label="RF" value={feature.stat_score} />
              <Score label="LASFS" value={feature.lasfs_score} />
              <Score label="Leakage" value={feature.leakage_risk} />
            </div>
            <p>{feature.reason}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function Score({ label, value }: { label: string; value: number }) {
  return (
    <span>
      {label}: {value.toFixed(3)}
    </span>
  );
}

function featureRowClass(feature: FeatureScore) {
  const severity =
    feature.leakage_risk > 0.8 ? 'risk-high' : feature.leakage_risk > 0.4 ? 'risk-medium' : 'risk-low';
  return ['feature-row', severity, feature.selected ? 'selected' : ''].filter(Boolean).join(' ');
}

async function parseJson(response: Response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = typeof payload.detail === 'string' ? payload.detail : 'Request failed.';
    throw new Error(message);
  }
  return payload;
}

export default App;
