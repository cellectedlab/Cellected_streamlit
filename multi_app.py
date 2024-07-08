import streamlit as st

# Import your Streamlit apps
import inflourescence_counter
import snp_analysis
import flask_cell_counter

def main():
    st.title('Multipage Streamlit App')

    # Navigation sidebar
    page = st.sidebar.selectbox(
        'Select App',
        ('Inflourescence Counter', 'SNP Analysis', 'Flask Cell Counter')
    )

    # Render the selected app
    if page == 'Inflourescence Counter':
        inflourescence_counter.main()
    elif page == 'SNP Analysis':
        snp_analysis.main()
    elif page == 'Flask Cell Counter':
        flask_cell_counter.main()

if __name__ == '__main__':
    main()
