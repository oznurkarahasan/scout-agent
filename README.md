## Scout Agent: Intelligent Real Estate Assistant

Scout Agent is an AI-powered real estate assistant that moves beyond rigid filtering systems. By combining Fuzzy Logic and Large Language Models (LLMs), it evaluates property listings with human-like reasoning to provide a personalized "Compatibility Score".

### Features

- Fuzzy Evaluation: Handles flexible criteria (e.g., "slightly expensive but great location") instead of strict binary filters.

- LLM-Powered Analysis: Uses Groq (Llama 3) to extract semantic meaning from property descriptions (e.g., balcony status, furniture quality).

- Multi-Criteria Scoring: Balances Price, Location, Listing Quality, Semantic Match, and Recency.

- Smart Notifications: Alerts users only when high-compatibility opportunities are found.

### Project Structure

```bash
.
├── src/
│   ├── fuzzy/      # Mamdani-type Fuzzy Inference System engine [cite: 45]
│   ├── llm/        # Groq API client and semantic analysis logic 
│   ├── data/       # Data loaders and synthetic dataset handlers [cite: 39]
│   └── utils/      # Notification and helper utilities 
├── data/           # Local synthetic JSON datasets
├── app.py          # Streamlit UI Entry Point
└── main.py         # CLI Entry Point

```

### Installation

1. Clone repo
```bash
git clone https://github.com/yourusername/scout-agent.git
cd scout-agent
```
2. Setup env
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment variables
Create a .env file in the root directory:
```bash
cp .env.example .env # On Windows copy .env.example .env
GROQ_API_KEY=your_groq_api_key_here
```

### Usage
Run in de cli:
```bash
python3 main.py
```
Run the Streamlit dashboard to interact with the assistant:
```bash
streamlit run app.py
```

## TODO

    [ ] Fuzzy Engine: Complete the Mamdani membership functions for all 5 inputs.

    [ ] Rule Base: Implement the 25-30 core logic rules.

    [ ] LLM Integration: Finalize the structured JSON extraction prompt for Groq.

    [ ] Dashboard: Create interactive sliders for user priority weighting.

    [ ] Notifications: Implement the email/push alert system for scores above a certain threshold.
