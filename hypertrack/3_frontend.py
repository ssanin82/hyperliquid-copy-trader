"""
Frontend Visualization - Stage 3: Display wallet profiling results.

Reads final_report.txt and creates a static HTML dashboard
with dark theme and grid layout similar to hypersignals.ai.
"""

import os
import re
import webbrowser
import json
from typing import Dict, Any

try:
    import plotly.graph_objects as go
except ImportError:
    print("Installing required packages: plotly")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import importlib
    importlib.invalidate_caches()
    import plotly.graph_objects as go

# Constants
FINAL_REPORT_FILE = "final_report.txt"
OUTPUT_HTML = "wallet_dashboard.html"


def parse_final_report(file_path: str) -> Dict[str, Any]:
    """Parse final_report.txt and extract wallet data."""
    wallets = []
    summary = {
        "generated": None,
        "total_wallets": 0,
        "wallets_with_categories": 0
    }
    
    if not os.path.exists(file_path):
        return {"summary": summary, "wallets": wallets}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse summary
    generated_match = re.search(r'Generated: (.+)', content)
    if generated_match:
        summary["generated"] = generated_match.group(1)
    
    total_match = re.search(r'Total wallets analyzed: (\d+)', content)
    if total_match:
        summary["total_wallets"] = int(total_match.group(1))
    
    categories_match = re.search(r'Wallets with categories: (\d+)', content)
    if categories_match:
        summary["wallets_with_categories"] = int(categories_match.group(1))
    
    # Parse wallets
    wallet_blocks = re.split(r'Wallet #\d+:', content)[1:]  # Skip header
    
    for block in wallet_blocks:
        wallet_data = {}
        
        # Extract address
        address_match = re.search(r'0x[a-fA-F0-9]{40}', block)
        if address_match:
            wallet_data["address"] = address_match.group(0)
        
        # Extract categories
        categories_match = re.search(r'Categories: (.+)', block)
        if categories_match:
            categories_str = categories_match.group(1).strip()
            if categories_str != "None":
                # Parse categories with confidence scores
                categories = []
                for cat_match in re.finditer(r'([^(]+)\(([\d.]+)\)', categories_str):
                    categories.append({
                        "name": cat_match.group(1).strip(),
                        "confidence": float(cat_match.group(2))
                    })
                wallet_data["categories"] = categories
            else:
                wallet_data["categories"] = []
        else:
            wallet_data["categories"] = []
        
        # Extract scores
        scores = {}
        score_patterns = {
            "trading_style": r"Trading Style Score \(6D vector\): \[(.*?)\]",
            "risk_score": r"Risk Score: ([\d.]+)",
            "profitability_score": r"Profitability Score: ([\d.]+)",
            "bot_probability": r"Bot Probability: ([\d.]+)",
            "influence_score": r"Influence Score: ([\d.]+)",
            "sophistication_score": r"Sophistication Score: ([\d.]+)",
            "transaction_count": r"Transaction Count: (\d+)"
        }
        
        for key, pattern in score_patterns.items():
            match = re.search(pattern, block)
            if match:
                if key == "trading_style":
                    # Parse vector
                    vector_str = match.group(1)
                    scores[key] = [float(x.strip().strip("'")) for x in vector_str.split(',')]
                else:
                    scores[key] = float(match.group(1)) if key != "transaction_count" else int(match.group(1))
        
        wallet_data["scores"] = scores
        
        if wallet_data.get("address"):
            wallets.append(wallet_data)
    
    return {"summary": summary, "wallets": wallets}


def create_score_gauge_dict(value: float, label: str, color: str, div_id: str) -> Dict[str, Any]:
    """Create dict data for a gauge chart using Plotly."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': label, 'font': {'size': 12, 'color': '#ffffff'}},
        gauge = {
            'axis': {'range': [None, 1], 'tickcolor': '#ffffff', 'tickfont': {'size': 10}},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 0.33], 'color': "rgba(255,255,255,0.1)"},
                {'range': [0.33, 0.66], 'color': "rgba(255,255,255,0.05)"},
                {'range': [0.66, 1], 'color': "rgba(255,255,255,0.05)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff', 'size': 10},
        height=120,
        margin=dict(l=5, r=5, t=25, b=5)
    )
    
    # Return dict representation (will be JSON encoded later)
    return fig.to_dict()


def create_wallet_card_html(wallet: Dict[str, Any], index: int) -> tuple:
    """Create HTML for a wallet card and return (html, gauge_data)."""
    address = wallet.get("address", "Unknown")
    categories = wallet.get("categories", [])
    scores = wallet.get("scores", {})
    
    # Shorten address for display
    address_short = f"{address[:6]}...{address[-4:]}" if len(address) > 10 else address
    
    # Category badges HTML
    # Filter out "Occasional Trader" category as it's not informative
    filtered_categories = [cat for cat in categories if cat.get("name", "").strip() != "Occasional Trader"]
    category_html = ""
    if filtered_categories:
        for cat in filtered_categories[:3]:  # Show top 3 categories
            confidence = cat.get("confidence", 0)
            if confidence > 0.5:
                badge_class = "badge-success"
            elif confidence > 0.3:
                badge_class = "badge-warning"
            else:
                badge_class = "badge-secondary"
            category_html += f'<span class="badge {badge_class} me-1 mb-1">{cat["name"]} ({confidence:.2f})</span>'
    else:
        category_html = '<span class="badge badge-secondary">No Categories</span>'
    
    # Score values
    risk_score = scores.get("risk_score", 0)
    profitability_score = scores.get("profitability_score", 0)
    bot_probability = scores.get("bot_probability", 0)
    sophistication_score = scores.get("sophistication_score", 0)
    transaction_count = scores.get("transaction_count", 0)
    
    # Create gauge charts dict data
    risk_gauge_dict = create_score_gauge_dict(risk_score, "Risk", "#ef4444", f"gauge-risk-{index}")
    profit_gauge_dict = create_score_gauge_dict(profitability_score, "Profit", "#10b981", f"gauge-profit-{index}")
    bot_gauge_dict = create_score_gauge_dict(bot_probability, "Bot", "#f59e0b", f"gauge-bot-{index}")
    soph_gauge_dict = create_score_gauge_dict(sophistication_score, "Soph", "#3b82f6", f"gauge-soph-{index}")
    
    card_html = f'''
    <div class="wallet-card">
        <div class="card-header">
            <div class="address-header">
                <h5 class="card-title">{address_short}</h5>
                <button class="copy-button" onclick="copyAddress('{address}', 'copy-btn-{index}')" id="copy-btn-{index}" title="Copy address">
                    <span class="copy-icon">⧉</span>
                </button>
            </div>
            <small class="text-muted address-full">{address}</small>
        </div>
        <div class="card-body">
            <div class="mb-3">
                <strong>Categories:</strong><br>
                <div class="mt-1">{category_html}</div>
            </div>
            <div class="mb-3">
                <strong>Transactions:</strong> <span class="text-info">{transaction_count}</span>
            </div>
            <div class="gauge-grid">
                <div class="gauge-item" id="gauge-risk-{index}"></div>
                <div class="gauge-item" id="gauge-profit-{index}"></div>
                <div class="gauge-item" id="gauge-bot-{index}"></div>
                <div class="gauge-item" id="gauge-soph-{index}"></div>
            </div>
        </div>
    </div>
    '''
    
    gauge_data = {
        f"gauge-risk-{index}": risk_gauge_dict,
        f"gauge-profit-{index}": profit_gauge_dict,
        f"gauge-bot-{index}": bot_gauge_dict,
        f"gauge-soph-{index}": soph_gauge_dict
    }
    
    return card_html, gauge_data


def generate_static_html(data: Dict[str, Any], output_path: str):
    """Generate a static HTML file with embedded CSS and JavaScript."""
    summary = data.get("summary", {})
    wallets = data.get("wallets", [])
    
    # Filter out wallets with no categories (excluding "Occasional Trader" which is not informative)
    filtered_wallets = []
    for wallet in wallets:
        categories = wallet.get("categories", [])
        # Filter out "Occasional Trader" and check if any meaningful categories remain
        meaningful_categories = [cat for cat in categories if cat.get("name", "").strip() != "Occasional Trader"]
        if meaningful_categories:  # Only include wallets with meaningful categories
            filtered_wallets.append(wallet)
    
    # Create wallet cards HTML and collect gauge data
    wallet_cards_html = ""
    all_gauge_data = {}
    for i, wallet in enumerate(filtered_wallets):
        card_html, gauge_data = create_wallet_card_html(wallet, i)
        wallet_cards_html += card_html
        all_gauge_data.update(gauge_data)
    
    # Read README.md if it exists
    readme_content = ""
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        # Try parent directory
        readme_path = os.path.join("..", "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
            # Escape for JavaScript
            readme_content = readme_content.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
    
    # Get Plotly.js and marked.js (we'll include them at the top)
    plotly_js = '''
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    '''
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wallet Profiling Dashboard</title>
    {plotly_js}
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
            color: #ffffff;
            min-height: 100vh;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header-section {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }}
        
        h1 {{
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0;
            flex: 1;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .docs-button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        
        .docs-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        }}
        
        .docs-button:active {{
            transform: translateY(0);
        }}
        
        .summary-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}
        
        .summary-card {{
            background: #1a1a2e;
            border: 1px solid #2d2d44;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .summary-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        }}
        
        .summary-card h3 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        .summary-card .text-primary {{
            color: #3b82f6;
        }}
        
        .summary-card .text-success {{
            color: #10b981;
        }}
        
        .summary-card .text-info {{
            color: #06b6d4;
        }}
        
        .summary-card p {{
            color: #9ca3af;
            font-size: 0.9rem;
        }}
        
        .wallet-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 1.5rem;
        }}
        
        .wallet-card {{
            background: #1a1a2e;
            border: 1px solid #2d2d44;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .wallet-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.5);
            border-color: #3b82f6;
        }}
        
        .card-header {{
            background: #16213e;
            padding: 1rem;
            border-bottom: 1px solid #2d2d44;
        }}
        
        .address-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.25rem;
        }}
        
        .card-title {{
            font-family: 'Courier New', monospace;
            font-size: 1.1rem;
            margin: 0;
            color: #ffffff;
            flex: 1;
        }}
        
        .address-full-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
        }}
        
        .address-full {{
            font-family: 'Courier New', monospace;
            font-size: 0.7rem;
            color: #6b7280;
            word-break: break-all;
            flex: 1;
        }}
        
        .copy-button {{
            background: rgba(59, 130, 246, 0.2);
            border: 1px solid #3b82f6;
            color: #3b82f6;
            padding: 0.4rem 0.6rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 32px;
            height: 32px;
        }}
        
        .copy-button:hover {{
            background: rgba(59, 130, 246, 0.3);
            transform: scale(1.05);
        }}
        
        .copy-button:active {{
            transform: scale(0.95);
        }}
        
        .copy-button.copied {{
            background: rgba(16, 185, 129, 0.2);
            border-color: #10b981;
            color: #10b981;
        }}
        
        .copy-button-small {{
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid #3b82f6;
            color: #3b82f6;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.7rem;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 24px;
            height: 24px;
            flex-shrink: 0;
        }}
        
        .copy-button-small:hover {{
            background: rgba(59, 130, 246, 0.25);
            transform: scale(1.1);
        }}
        
        .copy-button-small:active {{
            transform: scale(0.9);
        }}
        
        .copy-button-small.copied {{
            background: rgba(16, 185, 129, 0.2);
            border-color: #10b981;
            color: #10b981;
        }}
        
        .copy-icon {{
            font-size: 0.9rem;
            line-height: 1;
        }}
        
        .card-body {{
            padding: 1.5rem;
        }}
        
        .card-body strong {{
            color: #e5e7eb;
            display: block;
            margin-bottom: 0.5rem;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .badge-success {{
            background-color: #10b981;
            color: #ffffff;
        }}
        
        .badge-warning {{
            background-color: #f59e0b;
            color: #ffffff;
        }}
        
        .badge-secondary {{
            background-color: #6b7280;
            color: #ffffff;
        }}
        
        .text-info {{
            color: #06b6d4;
            font-weight: 600;
        }}
        
        .gauge-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.5rem;
            margin-top: 1rem;
        }}
        
        .gauge-item {{
            background: rgba(255,255,255,0.02);
            border-radius: 8px;
            padding: 0.5rem;
        }}
        
        /* Modal styles */
        .modal-overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            overflow-y: auto;
        }}
        
        .modal-overlay.active {{
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 2rem;
        }}
        
        .modal-content {{
            background: #1a1a2e;
            border: 1px solid #2d2d44;
            border-radius: 12px;
            max-width: 900px;
            width: 100%;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            position: relative;
        }}
        
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.5rem;
            border-bottom: 1px solid #2d2d44;
            position: sticky;
            top: 0;
            background: #1a1a2e;
            z-index: 10;
        }}
        
        .modal-header h2 {{
            margin: 0;
            color: #ffffff;
            font-size: 1.5rem;
        }}
        
        .modal-close {{
            background: transparent;
            border: 1px solid #2d2d44;
            color: #ffffff;
            width: 36px;
            height: 36px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }}
        
        .modal-close:hover {{
            background: #2d2d44;
        }}
        
        .modal-body {{
            padding: 2rem;
            color: #e5e7eb;
        }}
        
        .modal-body h1, .modal-body h2, .modal-body h3, .modal-body h4 {{
            color: #ffffff;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }}
        
        .modal-body h1 {{
            font-size: 2rem;
            border-bottom: 2px solid #2d2d44;
            padding-bottom: 0.5rem;
        }}
        
        .modal-body h2 {{
            font-size: 1.5rem;
            border-bottom: 1px solid #2d2d44;
            padding-bottom: 0.5rem;
        }}
        
        .modal-body h3 {{
            font-size: 1.25rem;
        }}
        
        .modal-body code {{
            background: #0f0f1e;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #10b981;
            border: 1px solid #2d2d44;
        }}
        
        .modal-body pre {{
            background: #0f0f1e;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid #2d2d44;
            margin: 1rem 0;
        }}
        
        .modal-body pre code {{
            background: transparent;
            padding: 0;
            border: none;
            color: #e5e7eb;
        }}
        
        .modal-body ul, .modal-body ol {{
            margin: 1rem 0;
            padding-left: 2rem;
        }}
        
        .modal-body li {{
            margin: 0.5rem 0;
        }}
        
        .modal-body a {{
            color: #3b82f6;
            text-decoration: none;
        }}
        
        .modal-body a:hover {{
            text-decoration: underline;
        }}
        
        .modal-body blockquote {{
            border-left: 4px solid #3b82f6;
            padding-left: 1rem;
            margin: 1rem 0;
            color: #9ca3af;
        }}
        
        .modal-body table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        .modal-body th, .modal-body td {{
            padding: 0.75rem;
            border: 1px solid #2d2d44;
            text-align: left;
        }}
        
        .modal-body th {{
            background: #0f0f1e;
            color: #ffffff;
            font-weight: 600;
        }}
        
        .modal-body tr:nth-child(even) {{
            background: rgba(255,255,255,0.02);
        }}
        
        @media (max-width: 768px) {{
            .wallet-grid {{
                grid-template-columns: 1fr;
            }}
            
            h1 {{
                font-size: 2rem;
            }}
            
            .header-section {{
                flex-direction: column;
                gap: 1rem;
            }}
            
            .modal-overlay.active {{
                padding: 1rem;
            }}
            
            .modal-content {{
                max-height: 95vh;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header-section">
            <h1>Wallet Profiling Dashboard</h1>
            <button class="docs-button" onclick="openDocs()">📚 Docs</button>
        </div>
        
        <div class="summary-section">
            <div class="summary-card">
                <h3 class="text-primary">{summary.get('total_wallets', 0)}</h3>
                <p>Total Wallets</p>
            </div>
            <div class="summary-card">
                <h3 class="text-success">{len(filtered_wallets)}</h3>
                <p>Wallets Displayed</p>
            </div>
            <div class="summary-card">
                <h3 class="text-info">{summary.get('generated', 'N/A')[:19] if summary.get('generated') else 'N/A'}</h3>
                <p>Generated</p>
            </div>
        </div>
        
        <div class="wallet-grid">
            {wallet_cards_html}
        </div>
    </div>
    
    <!-- Docs Modal -->
    <div id="docsModal" class="modal-overlay" onclick="closeDocsOnOverlay(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h2>Documentation</h2>
                <button class="modal-close" onclick="closeDocs()" aria-label="Close">×</button>
            </div>
            <div class="modal-body" id="docsContent"></div>
        </div>
    </div>
    
    <script>
        // README content
        const readmeContent = `{readme_content}`;
        
        // Render all gauge charts
        document.addEventListener('DOMContentLoaded', function() {{
            const gaugeData = {json.dumps(all_gauge_data, indent=None)};
            for (const [divId, chartData] of Object.entries(gaugeData)) {{
                if (chartData && chartData.data && chartData.layout) {{
                    Plotly.newPlot(divId, chartData.data, chartData.layout, {{displayModeBar: false}});
                }}
            }}
        }});
        
        // Docs modal functions
        function openDocs() {{
            const modal = document.getElementById('docsModal');
            const content = document.getElementById('docsContent');
            if (readmeContent) {{
                // Render markdown to HTML
                if (typeof marked !== 'undefined') {{
                    content.innerHTML = marked.parse(readmeContent);
                }} else {{
                    content.innerHTML = '<pre>' + readmeContent.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</pre>';
                }}
            }} else {{
                content.innerHTML = '<p>README.md not found.</p>';
            }}
            modal.classList.add('active');
            document.body.style.overflow = 'hidden';
        }}
        
        function closeDocs() {{
            const modal = document.getElementById('docsModal');
            modal.classList.remove('active');
            document.body.style.overflow = '';
        }}
        
        function closeDocsOnOverlay(event) {{
            if (event.target.id === 'docsModal') {{
                closeDocs();
            }}
        }}
        
        // Close modal on Escape key
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                closeDocs();
            }}
        }});
        
        // Copy address to clipboard
        function copyAddress(address, buttonId) {{
            navigator.clipboard.writeText(address).then(function() {{
                // Visual feedback
                const button = document.getElementById(buttonId);
                const originalClass = button.className;
                button.classList.add('copied');
                button.title = 'Copied!';
                
                // Reset after 2 seconds
                setTimeout(function() {{
                    button.classList.remove('copied');
                    button.title = 'Copy address';
                }}, 2000);
            }}).catch(function(err) {{
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = address;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                textArea.style.top = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                try {{
                    document.execCommand('copy');
                    const button = document.getElementById(buttonId);
                    button.classList.add('copied');
                    button.title = 'Copied!';
                    setTimeout(function() {{
                        button.classList.remove('copied');
                        button.title = 'Copy address';
                    }}, 2000);
                }} catch (err) {{
                    console.error('Failed to copy:', err);
                }}
                document.body.removeChild(textArea);
            }});
        }}
    </script>
</body>
</html>
    '''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Dashboard saved to: {os.path.abspath(output_path)}")


def main():
    """Main function to run the frontend."""
    # Get the report file path
    report_path = FINAL_REPORT_FILE
    if not os.path.exists(report_path):
        # Try in parent directory
        parent_report = os.path.join("..", FINAL_REPORT_FILE)
        if os.path.exists(parent_report):
            report_path = parent_report
        else:
            print(f"Error: {FINAL_REPORT_FILE} not found!")
            return
    
    print(f"Parsing {report_path}...")
    data = parse_final_report(report_path)
    
    if not data.get("wallets"):
        print("No wallets found in report!")
        return
    
    print(f"Found {len(data['wallets'])} wallets")
    print("Creating dashboard...")
    
    # Generate static HTML
    output_path = OUTPUT_HTML
    generate_static_html(data, output_path)
    
    # Open in browser
    abs_path = os.path.abspath(output_path)
    print(f"Opening {abs_path} in browser...")
    webbrowser.open(f"file://{abs_path}")
    
    print("Dashboard opened in browser!")


if __name__ == "__main__":
    main()
