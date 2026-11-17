import os
import json
import streamlit as st
import pandas as pd
import altair as alt
from openai import OpenAI

st.set_page_config(
    page_title="AI Therapy Progress Assistant",
    page_icon="🩺",
    layout="wide"
)

# Fictional demonstration records only.
data = pd.DataFrame([
    {"Patient ID":"DEMO-001","Session":1,"Date":"2026-10-01","Mobility score":42,"Pain score":7,"Exercise completion (%)":50,"Therapist note":"Needed support with balance exercises."},
    {"Patient ID":"DEMO-001","Session":2,"Date":"2026-10-08","Mobility score":47,"Pain score":6,"Exercise completion (%)":60,"Therapist note":"More confident with supported standing."},
    {"Patient ID":"DEMO-001","Session":3,"Date":"2026-10-15","Mobility score":51,"Pain score":5,"Exercise completion (%)":70,"Therapist note":"Completed more repetitions with fewer prompts."},
    {"Patient ID":"DEMO-001","Session":4,"Date":"2026-10-22","Mobility score":56,"Pain score":4,"Exercise completion (%)":75,"Therapist note":"Improved balance during supported walking."},
    {"Patient ID":"DEMO-002","Session":1,"Date":"2026-10-03","Mobility score":65,"Pain score":4,"Exercise completion (%)":80,"Therapist note":"Completed planned exercises."},
    {"Patient ID":"DEMO-002","Session":2,"Date":"2026-10-10","Mobility score":67,"Pain score":4,"Exercise completion (%)":85,"Therapist note":"Maintained performance across activities."},
    {"Patient ID":"DEMO-002","Session":3,"Date":"2026-10-17","Mobility score":69,"Pain score":3,"Exercise completion (%)":90,"Therapist note":"Home exercises felt manageable."},
    {"Patient ID":"DEMO-002","Session":4,"Date":"2026-10-24","Mobility score":71,"Pain score":3,"Exercise completion (%)":90,"Therapist note":"Small improvement in mobility score."},
])

st.title("🩺 AI Therapy Progress Assistant")
st.caption("A student prototype using Python, Streamlit and an AI language model.")
st.info(
    "DEMO ONLY: all records are fictional. "
    "This prototype is not a clinical decision-making tool."
)

# Patient selection
patient = st.sidebar.selectbox(
    "Choose demo patient",
    sorted(data["Patient ID"].unique())
)

df = data[data["Patient ID"] == patient].copy()
df["Date"] = pd.to_datetime(df["Date"])

latest = df.iloc[-1]
first = df.iloc[0]
previous = df.iloc[-2] if len(df) > 1 else latest

# Patient overview
st.subheader(f"Patient overview: {patient}")

a, b, c = st.columns(3)

a.metric(
    "Latest mobility score",
    f'{latest["Mobility score"]}/100',
    f'{int(latest["Mobility score"] - previous["Mobility score"])} since previous'
)

b.metric(
    "Latest pain score",
    f'{latest["Pain score"]}/10',
    f'{int(latest["Pain score"] - previous["Pain score"])} since previous',
    delta_color="inverse"
)

c.metric(
    "Exercise completion",
    f'{latest["Exercise completion (%)"]}%'
)

# Progress chart
st.subheader("Progress over time")

long_df = df.melt(
    id_vars=["Date", "Session"],
    value_vars=[
        "Mobility score",
        "Pain score",
        "Exercise completion (%)"
    ],
    var_name="Measure",
    value_name="Score"
)

chart = alt.Chart(long_df).mark_line(point=True).encode(
    x=alt.X("Date:T", title="Session date"),
    y=alt.Y("Score:Q", title="Recorded score"),
    color=alt.Color("Measure:N", title="Measure"),
    tooltip=["Date:T", "Session:Q", "Measure:N", "Score:Q"]
).properties(height=300)

st.altair_chart(chart, use_container_width=True)

# AI progress analysis
st.subheader("🤖 AI-generated progress analysis")

st.write(
    "Ask the AI model to interpret the recorded session data "
    "and produce a structured summary for therapist review."
)

if st.button("Generate AI analysis", type="primary"):

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error(
            "API key not found. Please set OPENAI_API_KEY "
            "in your Terminal and restart the app."
        )

    else:
        # Prepare the selected patient's records for the AI.
        records = df.copy()
        records["Date"] = records["Date"].dt.strftime("%Y-%m-%d")

        patient_data = records.to_dict(orient="records")

        prompt = f"""
        Analyse the following fictional rehabilitation session records.

        Patient: {patient}

        Session records:
        {json.dumps(patient_data, indent=2)}

        Please produce a clear report with these sections:

        1. Overall progress:
        Describe the trends in mobility, reported pain and exercise completion.

        2. Session observations:
        Identify meaningful changes across the sessions.

        3. Points for therapist review:
        Identify any patterns or questions that a therapist may wish to explore.

        4. Short summary:
        Write a concise paragraph summarising the recorded progress.

        Important limitations:
        - Only use the information provided.
        - Do not invent missing information.
        - Do not diagnose medical conditions.
        - Do not prescribe treatment or change exercises.
        - Do not assume that a higher mobility score or lower pain score
          necessarily means clinical improvement.
        - Clearly state that the report requires professional review.
        """

        try:
            with st.spinner("AI is analysing the session records..."):

                client = OpenAI(api_key=api_key)

                response = client.responses.create(
                    model="gpt-5.5",
                    instructions=(
                        "You are an assistant helping organise fictional "
                        "rehabilitation data for educational demonstration. "
                        "You are not a doctor or therapist. Be cautious, "
                        "factual and clear."
                    ),
                    input=prompt
                )

                st.session_state["ai_analysis"] = response.output_text
                st.session_state["ai_patient"] = patient

        except Exception as e:
            st.error(f"AI request failed: {e}")

# Display the most recent analysis for the selected patient.
if (
    st.session_state.get("ai_patient") == patient
    and st.session_state.get("ai_analysis")
):
    st.markdown("---")
    st.markdown(st.session_state["ai_analysis"])

    st.caption(
        "AI-generated educational summary. "
        "Not a diagnosis, treatment recommendation or clinical assessment."
    )

# Session history
st.subheader("Session history")
st.dataframe(df, use_container_width=True, hide_index=True)

# Add fictional session preview
st.subheader("Add a fictional demo session")

with st.form("new_session"):

    col1, col2, col3 = st.columns(3)

    with col1:
        session_date = st.date_input("Session date")
        mobility = st.slider("Mobility score", 0, 100, 60)

    with col2:
        pain = st.slider("Pain score", 0, 10, 3)
        completion = st.slider("Exercise completion (%)", 0, 100, 75)

    with col3:
        note = st.text_input(
            "Therapist note",
            "Completed planned exercises."
        )

    submit = st.form_submit_button("Preview session")

if submit:
    st.success(
        "Preview created. This starter version does not save "
        "entries after the app closes."
    )

    st.json({
        "Patient ID": patient,
        "Date": str(session_date),
        "Mobility score": mobility,
        "Pain score": pain,
        "Exercise completion (%)": completion,
        "Therapist note": note
    })

st.caption(
    "Student portfolio prototype | Python • Streamlit • Pandas "
    "• Altair • OpenAI API"
)
