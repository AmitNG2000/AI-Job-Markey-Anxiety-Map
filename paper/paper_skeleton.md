# AI Anxiety Map: Global Labor-Market Anxiety and Opportunity in the Age of Generative AI


## Authors
Amit Ner Gaon
The depratment of Software and Information Systems Engineering, 
Ben-Gurion University
amitner@post.bgu.ac.il

## Abstract

- Introduce the problem of understanding labor-market discourse around generative AI.
- State that the project uses GDELT Global Knowledge Graph data to compare discourse before and after a selected AI cutoff date.
- Summarize the main analysis dimensions: anxiety, opportunity, geography, temporal change, and network structure.
- Mention the expected contribution: a data-driven framework for measuring how labor-market perceptions shift across communities.
- End with a high-level summary of the main findings once the final results are available.

## 1. Introduction

- Present the rise of generative AI as a major social and economic event.
- Explain why labor-market reactions are important to study.
- Motivate the use of large-scale news data for measuring public discourse.
- Define the core idea of AI anxiety and contrast it with opportunity-related language.
- Introduce the before/after comparison around the AI cutoff date.
- Present the main research questions:
  - How do different communities perceive the labor market?
  - How do different communities perceive the impact of AI on employment?
  - How similar or polarized are labor-market perceptions across communities before and after generative AI?
- Summarize the paper structure.

## 2. Related Work

- Review prior work on AI, automation, and labor-market disruption.
- Discuss studies that analyze public sentiment, fear, uncertainty, or optimism about technological change.
- Introduce GDELT and its use in computational social science, news analysis, and global event monitoring.
- Cover sentiment and emotion analysis methods relevant to news text.
- Discuss network-based approaches for comparing communities, countries, or discourse patterns.
- Explain how this project connects these areas by combining sentiment, geography, time comparison, and network analysis.

## 3. Methodology

- Describe the overall project pipeline from GDELT collection to final analysis.
- Explain how labor-market-related records are selected from GDELT themes.
- Describe the extracted information:
  - article identifiers and dates;
  - locations;
  - labor-market themes;
  - GDELT tone values;
  - selected GCAM emotion dimensions.
- Define the before-AI and after-AI periods.
- Explain the construction of the main scores:
  - anxiety-related score;
  - opportunity-related score;
  - combined feeling score.
- Describe the aggregation level used in the analysis, such as location, time window, and theme.
- Explain the geographic comparison approach.
- Explain the network analysis approach:
  - location-emotion bipartite networks;
  - projected location-similarity networks;
  - difference networks between periods.
- Mention the tools and libraries used at a high level.

## 4. Experiments

### 4.1 Dataset

- Describe the GDELT Global Knowledge Graph as the data source.
- State the selected study period and AI cutoff date.
- Explain the use of main and translated GDELT streams if both are included.
- Describe the filtering process for labor-market themes.
- Summarize the final dataset in general terms:
  - number of records or interactions;
  - number of locations;
  - time coverage;
  - emotion and tone fields.
- Include a table describing the main dataset fields used in the analysis.

### 4.2 System Configuration

- Describe the computational environment used to run the pipeline.
- Mention the main Python packages used for data processing, plotting, and network analysis.
- Explain whether the work was run locally, on a cluster, or both.
- Describe the batch-processing setup for large GDELT downloads.
- Mention reproducibility elements such as configuration files, fixed date ranges, and saved output folders.

### 4.3 Results

- Present overall before/after changes in labor-market discourse.
- Show global changes in article volume, tone, anxiety, opportunity, and combined feeling score.
- Compare selected countries or communities.
- Present visual summaries:
  - period comparison plots;
  - per-location comparison plots;
  - dominant-feeling distributions;
  - network visualizations.
- Describe the location-emotion networks for each period.
- Describe the location-similarity networks and what they suggest about shared discourse patterns.
- Present changes in similarity or polarization between periods.
- Keep interpretation factual in this section and save broader meaning for the discussion.

## 5. Discussion

- Interpret the main patterns observed in the results.
- Discuss whether discourse becomes more anxious, more opportunity-oriented, or more mixed after generative AI.
- Explain differences between locations or communities.
- Discuss what the network structure suggests about shared perceptions and polarization.
- Connect the findings back to the original research questions.
- Discuss possible social, economic, or media-related explanations for the observed patterns.
- Explain why the results should be interpreted carefully.

### 5.1 Conclusion

- Restate the goal of the project.
- Summarize the main methodological contribution.
- Summarize the main empirical findings at a high level.
- Explain the value of combining GDELT, sentiment analysis, geography, and networks for studying AI-related labor discourse.
- End with a short statement about what the project contributes to the broader understanding of generative AI's societal impact.

### 5.2 Limitations

- Discuss limitations of using news data as a proxy for public perception.
- Mention possible bias in GDELT coverage across countries, languages, and media systems.
- Discuss limitations of keyword/theme filtering for identifying labor-market discourse.
- Explain that sentiment and emotion dictionaries may not fully capture context, sarcasm, or complex framing.
- Mention that location extraction may not always represent the affected community.
- Note that before/after comparisons do not prove causality.

### 5.3 Future Work

- Extend the study period and include more complete historical data.
- Improve AI-specific filtering to better separate general labor discourse from AI-related labor discourse.
- Compare different regions, income groups, or language groups.
- Test alternative sentiment and emotion models.
- Add topic modeling or clustering to identify major themes in AI labor-market discourse.
- Validate results against external data sources such as surveys, labor-market indicators, or policy events.

### 5.4 Acknowledgments

- Acknowledge the course staff and academic guidance.
- Acknowledge GDELT as the data source.
- Mention any computing resources used, such as university cluster resources.

## Appendix

## References