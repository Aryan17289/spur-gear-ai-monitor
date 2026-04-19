with tab3:

    # ---- CSS for chat bubbles ----
    st.markdown("""
    <style>
    .chat-user {
        background: #1e3a5f; color: white; border-radius: 16px 16px 4px 16px;
        padding: 12px 18px; margin: 8px 0 8px auto; max-width: 75%;
        font-size: 14px; line-height: 1.6; width: fit-content; margin-left: auto;
    }
    .chat-ai {
        background: white; color: #1e293b; border-radius: 16px 16px 16px 4px;
        padding: 12px 18px; margin: 8px auto 8px 0; max-width: 80%;
        font-size: 14px; line-height: 1.7; border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); width: fit-content;
    }
    .chat-label-user {
        text-align: right; font-size: 11px; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.06em;
        margin-bottom: 2px; margin-right: 4px;
    }
    .chat-label-ai {
        font-size: 11px; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.06em;
        margin-bottom: 2px; margin-left: 4px;
    }
    .briefing-box {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 16px 20px; font-size: 13px; color: #334155;
        font-family: 'DM Mono', monospace; line-height: 1.8; margin-bottom: 20px;
    }
    .briefing-title {
        font-size: 11px; color: #64748b; text-transform: uppercase;
        letter-spacing: 0.08em; font-weight: 600; margin-bottom: 10px;
        font-family: 'DM Sans', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- Build system briefing from live model outputs ----
    top_shap_feature = shap_df.sort_values("Impact", ascending=False).iloc[0]
    top_shap_stable  = shap_df.sort_values("Impact").iloc[0]
    top_lime_feat    = sorted(lime_list, key=lambda x: abs(x[1]), reverse=True)[0]

    briefing = f"""GEAR HEALTH BRIEFING — {datetime.now().strftime('%d %b %Y %H:%M')}
Gear Type        : {gear_type}
Failure Probability : {prob_pct:.1f}%
Risk Level       : {risk_label}
Prediction       : {"FAILURE DETECTED" if prediction == 1 else "NO FAILURE DETECTED"}

REMAINING USEFUL LIFE
Health Score     : {health_score * 100:.1f}% ({rul_label})
Est. RUL         : {rul_cycles:,.0f} cycles  (~{rul_hours:.1f} hrs at {speed} RPM)
RUL Range        : {rul_low:,.0f} – {rul_high:,.0f} cycles

CURRENT SENSOR READINGS
Speed            : {speed} RPM
Torque           : {torque} Nm
Vibration        : {vibration} mm/s
Temperature      : {temperature} °C
Shock Load       : {shock} g
Noise Level      : {noise} dB

SHAP ANALYSIS (top drivers)
Highest risk driver   : {top_shap_feature['Feature']} = {top_shap_feature['Value']} {top_shap_feature['Unit']}  (SHAP {top_shap_feature['Impact']:+.4f})
Greatest stabiliser   : {top_shap_stable['Feature']} = {top_shap_stable['Value']} {top_shap_stable['Unit']}  (SHAP {top_shap_stable['Impact']:+.4f})

LIME ANALYSIS (top local factor)
{top_lime_feat[0]}  (score {top_lime_feat[1]:+.4f})"""

    system_prompt = f"""You are an expert predictive maintenance AI assistant embedded in an industrial spur gear monitoring dashboard.

You have been given the following real-time gear health data from the machine learning models:

{briefing}

Your job is to answer questions from factory engineers clearly and concisely — in plain English, no jargon unless asked.
- Always ground your answers in the numbers above. Do not guess or make up values.
- Keep answers focused and practical. Engineers want to know what to DO.
- If asked something outside the scope of the data above, say so honestly.
- Format responses clearly. Use short paragraphs or numbered steps when helpful.
- Never mention that you are Claude or reference Anthropic."""

    # ---- Show live briefing ----
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 AI Maintenance Copilot")
    st.markdown("""
    <p style='font-size:14px;color:#475569;margin-top:-4px;margin-bottom:16px;line-height:1.65'>
        Ask the AI anything about the current gear health. It has full access to the live model outputs —
        failure probability, RUL estimate, SHAP values, and sensor readings.
        It will answer as a maintenance expert, grounded in your actual data.
        <strong>Updates automatically as you adjust the sliders.</strong>
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<div class='briefing-title'>Live Data Briefing fed to the AI</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='briefing-box'>{briefing.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ---- Chat interface ----
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("**Chat with the Copilot**")
    st.markdown("<p style='font-size:13px;color:#64748b;margin-top:0'>Ask questions about the current gear condition, what to inspect, how long you have, etc.</p>", unsafe_allow_html=True)

    # Initialise chat history
    if "copilot_messages" not in st.session_state:
        st.session_state.copilot_messages = []

    # Suggested questions
    st.markdown("<p style='font-size:12px;color:#94a3b8;margin-bottom:8px'>Suggested questions:</p>", unsafe_allow_html=True)
    sq1, sq2, sq3, sq4 = st.columns(4)
    suggestions = [
        "Why is this gear failing?",
        "How long do I have before shutdown?",
        "Which sensor should I check first?",
        "What maintenance should I do right now?",
    ]
    for col, suggestion in zip([sq1, sq2, sq3, sq4], suggestions):
        if col.button(suggestion, use_container_width=True):
            st.session_state.copilot_messages.append({"role": "user", "content": suggestion})

    # Display chat history
    if st.session_state.copilot_messages:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        for msg in st.session_state.copilot_messages:
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-label-user'>You</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-user'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-label-ai'>AI Copilot</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-ai'>{msg['content']}</div>", unsafe_allow_html=True)

    # Text input
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    user_input = st.text_input(
        "Your question",
        placeholder="e.g. Which parameter is most dangerous right now?",
        label_visibility="collapsed",
        key="copilot_input"
    )

    col_send, col_clear = st.columns([1, 5])
    send_clicked  = col_send.button("Send →", use_container_width=True)
    clear_clicked = col_clear.button("Clear chat", use_container_width=False)

    if clear_clicked:
        st.session_state.copilot_messages = []
        st.rerun()

    # Process message (either typed or suggestion button already appended)
    needs_response = (
        st.session_state.copilot_messages
        and st.session_state.copilot_messages[-1]["role"] == "user"
        and (len(st.session_state.copilot_messages) == 1
             or st.session_state.copilot_messages[-2]["role"] == "assistant"
             or send_clicked)
    )

    if send_clicked and user_input.strip():
        st.session_state.copilot_messages.append({"role": "user", "content": user_input.strip()})
        needs_response = True

    if needs_response:
        if not OPENAI_API_KEY:
            st.error("⚠ No API key found. Add OPENAI_API_KEY to your .env file.")
        else:
            with st.spinner("Thinking…"):
                try:
                    client = openai.OpenAI(
                        api_key=OPENAI_API_KEY,
                    )
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            *[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.copilot_messages
                            ]
                        ]
                    )
                    ai_reply = response.choices[0].message.content
                    st.session_state.copilot_messages.append(
                        {"role": "assistant", "content": ai_reply}
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"API error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)