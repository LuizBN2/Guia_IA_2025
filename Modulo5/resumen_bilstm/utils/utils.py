
import streamlit as st

# pie de página
def mostrar_firma_sidebar():
    st.sidebar.markdown("""
        <style>
            .firma-sidebar {
                position: fixed;
                bottom: 20px;
                left: 13px;
                width: 10pts;
                padding: 10px 15px;
                font-size: 0.8rem;
                border-radius: 10px;
                background-color: rgba(250, 250, 250, 0.9);
                z-index: 9999;
                text-align: left;
            }

            .firma-sidebar a {
                text-decoration: none;
                color: #333;
            }

            .firma-sidebar a:hover {
                color: #0077b5;
            }
        </style>

        <div class="firma-sidebar">
            Desarrollado por <strong>Mg. Luis Felipe Bustamante Narváez</strong><br>
            <a href="https://github.com/luizbn2" target="_blank">🐙 GitHub</a> · 
            <a href="https://www.linkedin.com/in/lfbn2" target="_blank">💼 LinkedIn</a>
        </div>
    """, unsafe_allow_html=True)