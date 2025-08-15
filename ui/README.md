# Op-Ed Analyzer UI

Streamlit web interface for analyzing op-ed documents. Provides an interactive dashboard for extracting claims, analyzing coherence, and fact-checking.

## Features Implemented

### Core Analysis Workflow
- **Configurable input**: Set number of documents (min 1, default 2) and claims per document (min 1, default 5)
- **Text paste interface**: Paste op-ed text directly into text areas (no file uploads)
- **Incremental loading**: Results display as soon as each step completes:
  1. Claims extraction → Show extracted claims immediately
  2. Coherence analysis → Show interactive matrix and metrics
  3. Fact checking → Show validation scores and explanations

### Interactive Visualizations (Plotly)
- **Coherence Matrix**: Interactive heatmap showing how claims affect each other's likelihood
  - Hover details with full claim text and reasoning
  - Color-coded relationships (red=negative, green=positive)
  - Document group separators
- **Validation Score Buckets**: Interactive scatter plot showing claim validation distribution
  - Color-coded buckets (red=low scores, green=high scores)
  - Hover details with claim text and explanations

### Interface Features
- **Document filtering**: Filter coherence and fact check results by document
- **Expandable sections**: Claims, coherence analysis, and fact checks can be expanded/collapsed
- **Session state management**: Analysis persists during session, navigate without losing progress
- **Cache hash display**: Shows cache hash for debugging and loading existing analyses
- **Debug loading**: Load previous analyses by cache hash
- **New Analysis button**: Reset to start fresh analysis

### Analysis Display
- **Claims section**: Shows extracted claims grouped by document
- **Coherence metrics**: Conflict prevalence, average intensity, max conflict
- **Load-bearing claims**: Top 3 claims with highest impact on others
- **Fact check summary**: Average validation score, most/least validated claims

## File Structure
```
ui/
├── app.py       # Main Streamlit application
├── utils.py     # UI utility functions for metrics calculation
└── README.md    # This documentation
```

## Running the UI

```bash
# From the oped directory
streamlit run ui/app.py
```

## Data Flow
1. Configure number of documents and claims per document
2. Paste op-ed text into document text areas
3. Click "Analyze Documents" (enabled when all documents have text)
4. Claims extraction runs and displays immediately
5. Coherence analysis runs and displays interactive matrix + metrics
6. Fact checking runs and displays validation buckets + summary
7. Use document filters to focus analysis on specific documents

## Technical Implementation
- **Framework**: Streamlit with custom CSS styling
- **Visualizations**: Plotly for interactive charts
- **State Management**: Streamlit session state for analysis persistence
- **Backend Integration**: Direct imports from core analysis modules
- **Caching**: Utilizes existing cache system with hash-based loading