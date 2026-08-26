import streamlit as st
import pandas as pd
import ollama

# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="Customer Insight AI",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# CUSTOM DESIGN
# -----------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

h1 {
    font-size: 42px !important;
    font-weight: 700 !important;
}

h2 {
    margin-top: 25px;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(128, 128, 128, 0.25);
    padding: 18px;
    border-radius: 12px;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.title("📊 Customer Insight AI")

    st.write(
        "Turn customer feedback into useful business insights using AI."
    )

    st.divider()

    st.subheader("How it works")

    st.write("1️⃣ Upload customer feedback")
    st.write("2️⃣ AI analyzes each customer")
    st.write("3️⃣ View business insights")
    st.write("4️⃣ Download the report")

    st.divider()

    st.caption("Powered by Ollama • Local AI")


# -----------------------------
# HEADER
# -----------------------------

st.title("📊 Customer Insight AI")

st.markdown(
    "### Turn customer feedback into actionable business insights."
)

st.write(
    "Upload a CSV containing customer feedback and let AI identify "
    "sentiment, problems, and recommended actions."
)

st.divider()

# -----------------------------
# FILE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Customer Feedback CSV",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    # -----------------------------
    # CHECK CSV
    # -----------------------------

    if "Customer" not in data.columns or "Feedback" not in data.columns:

        st.error(
            "❌ Your CSV must contain two columns: Customer and Feedback"
        )

    else:

        st.success(
            f"✅ File uploaded successfully — {len(data)} customers found."
        )

        # -----------------------------
        # ORIGINAL DATA
        # -----------------------------

        with st.expander("📄 View Uploaded Feedback"):

            st.dataframe(
                data,
                use_container_width=True
            )

        st.divider()

        # -----------------------------
        # ANALYZE BUTTON
        # -----------------------------

        if st.button(
            "🚀 Analyze Customer Feedback",
            type="primary"
        ):

            results = []

            progress = st.progress(0)

            status = st.empty()

            for index, row in data.iterrows():

                customer = row["Customer"]
                feedback = row["Feedback"]

                status.write(
                    f"🤖 Analyzing {customer} "
                    f"({index + 1}/{len(data)})..."
                )

                messages = [

                    {
                        "role": "system",

                        "content": """
You are a professional customer feedback analyzer.

Return EXACTLY these three lines:

Sentiment: Positive or Negative
Main issue: short phrase
Recommended action: short business action

Rules:
- Sentiment must be ONLY Positive or Negative.
- If there is no problem, write Main issue: None.
- If there is no problem, write Recommended action: No action required.
- Keep the main issue short.
- Keep the recommended action short.
- Use only information from the customer's feedback.
- Do not invent information.
- Do not explain anything.
- Do not add extra text.
"""
                    },

                    {
                        "role": "user",

                        "content":
                        f"Customer feedback: {feedback}"
                    }

                ]

                try:

                    response = ollama.chat(

                        model="phi3:latest",

                        messages=messages,

                        options={
                            "temperature": 0,
                            "num_predict": 50
                        }
                    )

                    analysis = response[
                        "message"
                    ]["content"].strip()

                except Exception:

                    analysis = (
                        "Sentiment: Unknown\n"
                        "Main issue: Analysis failed\n"
                        "Recommended action: Try again"
                    )

                # -----------------------------
                # DEFAULT VALUES
                # -----------------------------

                sentiment = "Unknown"
                main_issue = "Unknown"
                recommendation = "Unknown"

                # -----------------------------
                # READ AI RESPONSE
                # -----------------------------

                for line in analysis.splitlines():

                    line = line.strip()

                    if line.startswith("Sentiment:"):

                        sentiment = (
                            line
                            .replace("Sentiment:", "")
                            .strip()
                        )

                    elif line.startswith("Main issue:"):

                        main_issue = (
                            line
                            .replace("Main issue:", "")
                            .strip()
                        )

                    elif line.startswith(
                        "Recommended action:"
                    ):

                        recommendation = (
                            line
                            .replace(
                                "Recommended action:",
                                ""
                            )
                            .strip()
                        )

                # -----------------------------
                # SAVE RESULT
                # -----------------------------

                results.append({

                    "Customer": customer,

                    "Feedback": feedback,

                    "Sentiment": sentiment,

                    "Main Issue": main_issue,

                    "Recommended Action": recommendation

                })

                progress.progress(
                    (index + 1) / len(data)
                )

            progress.empty()
            status.empty()

            output = pd.DataFrame(results)

            st.success(
                "🎉 Analysis completed successfully!"
            )

            # -----------------------------
            # DASHBOARD
            # -----------------------------

            st.header("📊 Business Dashboard")

            total = len(output)

            positive = (
                output["Sentiment"] == "Positive"
            ).sum()

            negative = (
                output["Sentiment"] == "Negative"
            ).sum()

            if total > 0:

                positive_percent = (
                    positive / total
                ) * 100

                negative_percent = (
                    negative / total
                ) * 100

            else:

                positive_percent = 0
                negative_percent = 0

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "👥 Total Customers",
                total
            )

            col2.metric(
                "😊 Positive",
                positive
            )

            col3.metric(
                "⚠️ Negative",
                negative
            )

            col4.metric(
                "📈 Positive %",
                f"{positive_percent:.1f}%"
            )

            # -----------------------------
            # SENTIMENT
            # -----------------------------

            st.subheader("😊 Sentiment Overview")

            sentiment_counts = (
                output["Sentiment"]
                .value_counts()
            )

            st.bar_chart(
                sentiment_counts
            )

            # -----------------------------
            # ISSUES
            # -----------------------------

            st.subheader(
                "⚠️ Most Common Customer Issues"
            )

            issues = output[
                (output["Main Issue"] != "None") &
                (output["Main Issue"] != "Unknown") &
                (output["Main Issue"] != "Analysis failed")
            ]

            if not issues.empty:

                issue_counts = (
                    issues["Main Issue"]
                    .value_counts()
                    .head(10)
                )

                st.bar_chart(
                    issue_counts
                )

                most_common_issue = (
                    issue_counts.index[0]
                )

                st.info(
                    f"🔎 Most frequently reported issue: "
                    f"**{most_common_issue}**"
                )

            else:

                st.success(
                    "🎉 No major issues were identified."
                )

            # -----------------------------
            # SEARCH & FILTER
            # -----------------------------

            st.subheader(
                "🔎 Explore Customer Feedback"
            )

            search = st.text_input(
                "Search customer or feedback"
            )

            sentiment_filter = st.selectbox(
                "Filter by sentiment",
                [
                    "All",
                    "Positive",
                    "Negative"
                ]
            )

            filtered = output.copy()

            if search:

                filtered = filtered[
                    filtered["Customer"]
                    .astype(str)
                    .str.contains(
                        search,
                        case=False,
                        na=False
                    )
                    |
                    filtered["Feedback"]
                    .astype(str)
                    .str.contains(
                        search,
                        case=False,
                        na=False
                    )
                ]

            if sentiment_filter != "All":

                filtered = filtered[
                    filtered["Sentiment"]
                    == sentiment_filter
                ]

            st.dataframe(
                filtered,
                use_container_width=True
            )

            # -----------------------------
            # AI BUSINESS SUMMARY
            # -----------------------------

            st.subheader(
                "🧠 Overall Business Insight"
            )

            summary_text = f"""
Customer feedback analysis:

Total customers: {total}
Positive feedback: {positive}
Negative feedback: {negative}

Main issues:
{output["Main Issue"].to_string(index=False)}

Recommended actions:
{output["Recommended Action"].to_string(index=False)}

Provide a short business summary with:
1. What customers like
2. Biggest problems
3. What the business should prioritize

Keep it concise and practical.
"""

            try:

                summary_response = ollama.chat(

                    model="phi3:latest",

                    messages=[
                        {
                            "role": "user",
                            "content": summary_text
                        }
                    ],

                    options={
                        "temperature": 0,
                        "num_predict": 200
                    }
                )

                summary = summary_response[
                    "message"
                ]["content"].strip()

                st.info(summary)

            except Exception:

                st.warning(
                    "Business summary could not be generated."
                )

            # -----------------------------
            # DOWNLOAD
            # -----------------------------

            st.subheader(
                "📥 Download Your Report"
            )

            csv = output.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(

                label="📥 Download Complete Analysis",

                data=csv,

                file_name="customer_analysis.csv",

                mime="text/csv"
            )

else:

    st.info(
        "👆 Upload a CSV file above to begin."
    )

    st.markdown(
        """
        **Your CSV should contain:**

        `Customer` | `Feedback`

        Example:

        `Ali` | `The phone battery does not last long.`
        """
    )