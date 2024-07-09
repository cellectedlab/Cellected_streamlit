import streamlit as st

# Import your Streamlit apps
import inflourescence_counter
import snp_analysis
import flask_cell_counter
import snp_streamlit_version_2

def main():
    st.title('Multipage Streamlit App')

    # Navigation sidebar
    page = st.sidebar.selectbox(
        'Select App',
        ('Inflourescence Counter', 'SNP Analysis', 'Flask Cell Counter', 'SNP combined analysis')
    )

    # Add a warning message
    st.sidebar.warning('Changing pages will reset the data. Download results before switching.')

    # Render the selected app
    if page == 'Inflourescence Counter':
        inflourescence_counter.main()
    elif page == 'SNP Analysis':
        snp_analysis.main()
    elif page == 'Flask Cell Counter':
        flask_cell_counter.main()
    elif page == 'SNP combined analysis':
        snp_streamlit_version_2.main()

if __name__ == '__main__':
    main()
