import { useEffect, useMemo, useState } from "react";
import axios from "axios";

const API_BASE = "http://localhost:8000";

const emptyRows = [];

function parseCsvList(value) {
  return value
    .split(",")
    .map((v) => v.trim())
    .filter(Boolean);
}

function parseNumberList(value) {
  const items = parseCsvList(value);
  const nums = items.map((v) => Number(v));
  if (nums.some((n) => Number.isNaN(n))) return null;
  return nums;
}

export default function App() {
  const [itemType, setItemType] = useState("all");
  const [k, setK] = useState(5);
  const [alpha, setAlpha] = useState(0.6);
  const [diversityLambda, setDiversityLambda] = useState(1.0);
  const [recencyHalfLife, setRecencyHalfLife] = useState(30);
  const [mixTypes, setMixTypes] = useState(false);
  const [items, setItems] = useState([]);
  const [users, setUsers] = useState([]);
  const [itemId, setItemId] = useState("");
  const [userId, setUserId] = useState("");
  const [query, setQuery] = useState("");
  const [historyItems, setHistoryItems] = useState("P1, M1");
  const [historyWeights, setHistoryWeights] = useState("");
  const [activeTab, setActiveTab] = useState("item");
  const [results, setResults] = useState(emptyRows);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let ignore = false;
    async function load() {
      try {
        const [itemsRes, usersRes] = await Promise.all([
          axios.get(`${API_BASE}/items?item_type=all`),
          axios.get(`${API_BASE}/users`)
        ]);
        if (ignore) return;
        setItems(itemsRes.data.items || []);
        setUsers(usersRes.data.users || []);
      } catch (err) {
        if (!ignore) setError("Failed to load data from API.");
      }
    }
    load();
    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    if (items.length && !itemId) setItemId(items[0].item_id);
  }, [items, itemId]);

  useEffect(() => {
    if (users.length && !userId) setUserId(users[0]);
  }, [users, userId]);

  const filteredItems = useMemo(() => {
    if (itemType === "all") return items;
    return items.filter((item) => item.type === itemType);
  }, [items, itemType]);

  async function handleRecommend() {
    setError("");
    setLoading(true);
    try {
      let res;
      if (activeTab === "item") {
        res = await axios.post(`${API_BASE}/recommend/item`, {
          item_id: itemId,
          k,
          item_type: itemType,
          diversity_lambda: diversityLambda,
          mix_types: mixTypes
        });
      } else if (activeTab === "user") {
        res = await axios.post(`${API_BASE}/recommend/user`, {
          user_id: userId,
          k,
          item_type: itemType,
          alpha,
          recency_half_life_days: recencyHalfLife > 0 ? recencyHalfLife : null,
          diversity_lambda: diversityLambda,
          mix_types: mixTypes
        });
      } else if (activeTab === "history") {
        const itemsList = parseCsvList(historyItems);
        const weightsList = historyWeights.trim()
          ? parseNumberList(historyWeights)
          : null;

        res = await axios.post(`${API_BASE}/recommend/history`, {
          items: itemsList,
          weights: weightsList,
          k,
          item_type: itemType,
          alpha,
          diversity_lambda: diversityLambda,
          mix_types: mixTypes
        });
      } else if (activeTab === "text") {
        res = await axios.post(`${API_BASE}/recommend/text`, {
          text: query,
          k,
          item_type: itemType,
          diversity_lambda: diversityLambda,
          mix_types: mixTypes
        });
      } else {
        res = await axios.post(`${API_BASE}/recommend/popular`, {
          k,
          item_type: itemType
        });
      }
      setResults(res.data.items || emptyRows);
    } catch (err) {
      setError("Recommendation request failed. Is the API running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <div className="hero-text">
          <p className="eyebrow">Hybrid recommendations for products + music</p>
          <h1>Recommendation Engine</h1>
          <p className="subtext">
            Blend content signals with collaborative signals to surface better
            matches.
          </p>
        </div>
        <div className="card">
          <h3>Filters</h3>
          <label>
            Item type
            <select value={itemType} onChange={(e) => setItemType(e.target.value)}>
              <option value="all">All</option>
              <option value="products">Products</option>
              <option value="music">Music</option>
            </select>
          </label>
          <label>
            Results (k)
            <input
              type="number"
              min="3"
              max="10"
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
            />
          </label>
          <label>
            Alpha (content weight)
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={alpha}
              onChange={(e) => setAlpha(Number(e.target.value))}
            />
          </label>
          <div className="alpha-value">{alpha.toFixed(2)}</div>
          <label>
            Diversity (higher = more similar)
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={diversityLambda}
              onChange={(e) => setDiversityLambda(Number(e.target.value))}
            />
          </label>
          <div className="alpha-value">{diversityLambda.toFixed(2)}</div>
          <label>
            Recency half-life (days)
            <input
              type="number"
              min="0"
              max="365"
              value={recencyHalfLife}
              onChange={(e) => setRecencyHalfLife(Number(e.target.value))}
            />
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={mixTypes}
              onChange={(e) => setMixTypes(e.target.checked)}
            />
            Mix products + music
          </label>
        </div>
      </header>

      <section className="panel">
        <div className="tabs">
          <button
            className={activeTab === "item" ? "active" : ""}
            onClick={() => setActiveTab("item")}
          >
            By Item
          </button>
          <button
            className={activeTab === "user" ? "active" : ""}
            onClick={() => setActiveTab("user")}
          >
            By User
          </button>
          <button
            className={activeTab === "history" ? "active" : ""}
            onClick={() => setActiveTab("history")}
          >
            By History
          </button>
          <button
            className={activeTab === "text" ? "active" : ""}
            onClick={() => setActiveTab("text")}
          >
            By Text
          </button>
          <button
            className={activeTab === "popular" ? "active" : ""}
            onClick={() => setActiveTab("popular")}
          >
            Popular
          </button>
        </div>

        <div className="controls">
          {activeTab === "item" && (
            <label>
              Choose item
              <select value={itemId} onChange={(e) => setItemId(e.target.value)}>
                {filteredItems.map((item) => (
                  <option key={item.item_id} value={item.item_id}>
                    {item.item_id} - {item.title}
                  </option>
                ))}
              </select>
            </label>
          )}

          {activeTab === "user" && (
            <label>
              Choose user
              <select value={userId} onChange={(e) => setUserId(e.target.value)}>
                {users.map((user) => (
                  <option key={user} value={user}>
                    {user}
                  </option>
                ))}
              </select>
            </label>
          )}

          {activeTab === "history" && (
            <>
              <label>
                History items (comma-separated)
                <input
                  type="text"
                  value={historyItems}
                  placeholder="P1, M1, P3"
                  onChange={(e) => setHistoryItems(e.target.value)}
                />
              </label>
              <label>
                Optional weights (comma-separated)
                <input
                  type="text"
                  value={historyWeights}
                  placeholder="5, 3, 4"
                  onChange={(e) => setHistoryWeights(e.target.value)}
                />
              </label>
            </>
          )}

          {activeTab === "text" && (
            <label>
              Describe what you want
              <input
                type="text"
                value={query}
                placeholder="wireless bass headphones"
                onChange={(e) => setQuery(e.target.value)}
              />
            </label>
          )}

          {activeTab === "popular" && (
            <p className="hint">Shows highest-rated items from interactions.</p>
          )}

          <button className="cta" onClick={handleRecommend} disabled={loading}>
            {loading ? "Working..." : "Recommend"}
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        <div className="results">
          {results.length === 0 ? (
            <div className="empty">No results yet. Try a recommendation.</div>
          ) : (
            results.map((item) => (
              <div key={item.item_id} className="result-card">
                <h4>{item.title}</h4>
                <p className="meta">
                  <span>{item.type}</span>
                  <span>Score: {Number(item.score).toFixed(3)}</span>
                </p>
                <p className="tags">
                  {(item.tags || item.genre || item.category || "")
                    .toString()
                    .split(",")
                    .slice(0, 4)
                    .join(" · ")}
                </p>
              </div>
            ))
          )}
        </div>
      </section>

      <section className="panel">
        <h3>Item Preview</h3>
        <div className="grid">
          {filteredItems.slice(0, 6).map((item) => (
            <div key={item.item_id} className="mini-card">
              <strong>{item.title}</strong>
              <span>{item.type}</span>
              <span className="muted">{item.brand || item.artist || ""}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
