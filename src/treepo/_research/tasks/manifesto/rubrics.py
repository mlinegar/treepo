"""
Rubrics for RILE-preserving summarization.

These rubrics instruct the OPS system on what information to preserve
when summarizing political manifestos for left-right position scoring.
"""

# Task context for RILE scoring - explains the scoring task to the LLM
RILE_TASK_CONTEXT = """
Task: Score this political text on the left-right (RILE) scale.

The RILE (Right-Left) scale measures political position based on emphasis
on different policy domains. The scale ranges from -100 (far left) to +100 (far right).

LEFT indicators (negative score contributions):
- Decolonization, anti-imperialism
- Military: negative, peace emphasis
- Internationalism: positive
- Market regulation emphasis
- Economic planning emphasis
- Welfare state expansion
- Education expansion
- Labour groups: positive

RIGHT indicators (positive score contributions):
- Foreign special relationships: positive
- Military: positive
- Freedom and human rights (traditional interpretation)
- Free enterprise emphasis
- Economic incentives
- Protectionism: negative (free trade)
- Welfare state limitation
- National way of life: positive
- Traditional morality: positive
- Law and order emphasis
- Civic mindedness

Score range: -100 (far left) to +100 (far right)
A score of 0 indicates a centrist position with balanced left/right emphasis.

Output requirements:
- Return exactly one numeric score in the range [-100, +100].
- Estimate as precisely as possible on the continuous scale (avoid coarse buckets/threshold bins).
- If document metadata (year/country/party) is provided, use it only as contextual background.
- Formatting examples (do not copy; compute the actual score): -12, 0, 37.5
- Do not return reasoning, labels (e.g., "Score:"), markdown/backticks/code fences, or extra text.
"""


# Rubric for OPS summarization - tells the system what to preserve
RILE_PRESERVATION_RUBRIC = """
Task: This text will be scored on a left-right political scale (RILE).

Preserve ALL information relevant to determining left-right position:

LEFT indicators to preserve:
- Anti-imperialism, decolonization statements
- Peace emphasis, anti-military statements
- Internationalism, international cooperation
- Market regulation, government intervention in markets
- Economic planning, state control of economy
- Nationalization proposals
- Welfare state expansion (social security, healthcare, pensions)
- Education expansion emphasis
- Labour/union support statements
- Equality and redistribution statements
- Environmental protection

RIGHT indicators to preserve:
- Free enterprise, market economy statements
- Economic incentives, tax cuts for businesses
- Protectionism negative (pro free trade)
- Welfare state limitation statements
- Military positive, defense spending support
- National pride, traditional way of life statements
- Traditional morality, family values
- Law and order emphasis, tough on crime
- Civic mindedness, individual responsibility

Also preserve:
- Specific policy commitments with numbers
- Intensity of positions (strong vs weak statements)
- Relative emphasis (how much space devoted to each topic)
- Explicit party positioning statements ("we believe", "we reject")

DO NOT lose:
- Concrete policy proposals that indicate left/right positions
- Statements about the role of government vs market
- Social policy positions (welfare, education, healthcare)
- Economic policy positions (taxation, regulation, planning)
- Security and military positions
"""


# Alternative rubric focusing on economic dimension only
ECONOMIC_RUBRIC = """
Task: This text will be scored on economic left-right position.

Preserve information about economic policy stance:

LEFT ECONOMIC indicators:
- State intervention in the economy
- Nationalization of industries
- Price controls and market regulation
- Progressive taxation
- Redistribution policies
- Public sector expansion
- Workers' rights and union support

RIGHT ECONOMIC indicators:
- Free market emphasis
- Privatization
- Deregulation
- Tax cuts (especially corporate)
- Limited government spending
- Property rights protection
- Business-friendly policies

Preserve specific numbers, percentages, and concrete policy proposals.
"""


# Alternative rubric focusing on social/cultural dimension
SOCIAL_RUBRIC = """
Task: This text will be scored on social/cultural left-right position.

Preserve information about social and cultural policy stance:

LEFT SOCIAL indicators:
- Multiculturalism support
- Immigration positive
- Minority rights
- Gender equality
- LGBTQ+ rights
- Secular state
- Environmental protection
- International human rights

RIGHT SOCIAL indicators:
- Traditional values
- National identity emphasis
- Immigration skepticism/restriction
- Law and order emphasis
- Traditional family values
- Religious/moral traditionalism
- National sovereignty

Preserve statements about values, identity, and social norms.
"""


IMMIGRATION_RUBRIC = """
Task: This text will be scored on immigration policy stance (1 = strongly opposes
tough policy, 7 = strongly favors tough policy).

Preserve all information about immigration policy:
- Border control, deportations, asylum policy
- Quotas / caps / points-based systems
- Refugee acceptance or rejection
- Integration vs assimilation language
- Citizenship, naturalization
- Treatment of undocumented migrants
- Statements about cultural impact of migration
- EU free-movement / Schengen positions
Preserve concrete proposals (numbers, mechanisms) and rhetorical framing.
"""


EU_RUBRIC = """
Task: This text will be scored on orientation toward European integration
(1 = strongly opposed, 7 = strongly in favor).

Preserve all information about the party's stance on the EU:
- Support for or opposition to EU membership
- Position on deeper integration vs national sovereignty
- Eurozone / single currency stance
- Common foreign / defense / migration policy
- EU institutions (Commission, Parliament, ECJ) — accept or reform?
- Treaty changes, Brexit-style positions
- Trade with EU / customs union / single market
Preserve concrete proposals and the framing (pro-EU, EU-skeptic, federalist, etc.).
"""


ENVIRONMENT_RUBRIC = """
Task: This text will be scored on environment vs growth tradeoff (1 = environment
even at cost of growth, 7 = growth even at cost of environment).

Preserve all information about environmental and energy policy:
- Climate change framing and targets (net-zero dates, emission reductions)
- Renewable vs fossil-fuel energy positions
- Environmental regulation (industrial, transport, agricultural)
- Natural-resource protection, biodiversity
- Carbon pricing, green taxes, subsidies
- Trade-offs against jobs, growth, competitiveness
- Just-transition language
Preserve concrete proposals and the explicit balance struck against economic concerns.
"""


DECENTRALIZATION_RUBRIC = """
Task: This text will be scored on political decentralization (1 = strongly favors
decentralization to regions/localities, 7 = strongly opposes decentralization).

Preserve all information about the territorial organization of the state:
- Regional / federal / local government powers
- Devolution proposals (transfer of competencies)
- Fiscal autonomy of regions
- Recognition of national minorities, regional languages
- Subsidiarity language
- Independence movements; federalism / unitarism positions
- Centralization of decision-making vs local autonomy
Preserve concrete proposals about what decisions are made at which level.
"""


JOINT_RUBRIC = """
Task: This text will be scored on SIX policy dimensions, all on a 1-7 scale.
Produce one summary that preserves evidence relevant to every dimension.

For EACH of the following, preserve concrete policy proposals, rhetorical
stance, specific numbers or targets, and the party's relative emphasis:

1. ECONOMIC (public services vs tax reduction):
   - Taxation, public spending, welfare, nationalization vs privatization,
     regulation vs deregulation, redistribution, labor policy.

2. SOCIAL / LIBERALISM (liberal values vs traditional values):
   - Gender equality, LGBTQ+ rights, minority rights, religious / moral
     positions, family policy, multiculturalism vs traditionalism.

3. IMMIGRATION (permissive vs tough):
   - Border control, asylum, refugee policy, quotas, integration vs
     assimilation, citizenship, cultural-impact framing.

4. EU INTEGRATION (opposed vs in favor):
   - EU membership, deeper integration vs sovereignty, Eurozone, treaty
     changes, trade with EU, EU institutions.

5. ENVIRONMENT (environmental protection vs economic growth):
   - Climate targets, renewables vs fossil fuels, regulation, carbon
     pricing, growth/jobs tradeoffs, just-transition language.

6. DECENTRALIZATION (favor vs oppose):
   - Regional / federal / local powers, devolution, fiscal autonomy,
     minority / regional recognition, federalism vs unitarism.

If the text is silent on a dimension, say so explicitly for that dimension
rather than inventing content. Preserve the original wording of key
quotes where possible.
"""
