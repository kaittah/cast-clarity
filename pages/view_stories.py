import os
import json

import modal
import streamlit as st

def process_podcast_info(url):
    f = modal.Function.lookup("corise-podcast-project", "process_podcast")
    output = f.call(url, '/content/podcast/')
    return output

def create_dict_from_json_files(folder_path):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    data_dict = {}

    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            podcast_info = json.load(file)
            podcast_name = podcast_info['podcast_details']['podcast_title']
            episode_name = podcast_info['podcast_details']['episode_title']
            data_dict[f"{podcast_name} - {episode_name}"] = podcast_info

    return data_dict

def set_bg():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://img.freepik.com/free-vector/cloud-background-vector-cute-desktop-wallpaper_53876-136885.jpg?w=1800&t=st=1692448859~exp=1692449459~hmac=70658c3526a7bf6f5b4cc250a1c017152c2d47ccd4541b7c33153a997030f4f5");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def main():
    set_bg()
    st.title("View Stories")

    available_podcast_info = create_dict_from_json_files('processed_podcasts')

    # Left section - Input fields
    st.sidebar.header("Podcast RSS Feeds")

    # Dropdown box
    st.sidebar.subheader("Available Podcasts Feeds")
    selected_podcast = st.sidebar.selectbox("Select Podcast", options=available_podcast_info.keys())

    if selected_podcast:

        podcast_info = available_podcast_info[selected_podcast]

        # Right section - Newsletter content
        st.header("Newsletter Content")

        # Display the podcast title
        st.subheader("Episode Title")
        st.write(podcast_info['podcast_details']['episode_title'])

        # Display the podcast summary and the cover image in a side-by-side layout
        col1, col2 = st.columns([7, 3])

        with col1:
            # Display the podcast episode summary
            st.subheader("Podcast Episode Summary")
            st.write(podcast_info['podcast_summary'])

        with col2:
            st.image(podcast_info['podcast_details']['episode_image'], caption="Podcast Cover", width=300, use_column_width=True)

        # Display the podcast guest and their details in a side-by-side layout
        col3, col4 = st.columns([3, 7])

        with col3:
            st.subheader("Podcast Guest")
            st.write(podcast_info['podcast_guest']['Guest'])
            if podcast_info['podcast_guest'].get('Title'):
                st.write('Title: ' + podcast_info['podcast_guest']['Title'])
            if podcast_info['podcast_guest'].get('Organization'):
                st.write('Organization: ' + podcast_info['podcast_guest']['Organization'])
        
        with col4:
            if podcast_info['podcast_guest'].get('Wiki Summary'):
                st.write(podcast_info['podcast_guest']['Wiki Summary'])


        # Display the key moments
        st.subheader("Key Moments")
        key_moments = podcast_info['podcast_highlights']
        for moment in key_moments.split('\n'):
            st.markdown(
                f"<p style='margin-bottom: 5px;'>{moment}</p>", unsafe_allow_html=True)

    # User Input box
    st.sidebar.subheader("Add and Process New Podcast Feed")
    url = st.sidebar.text_input("Link to RSS Feed")

    process_button = st.sidebar.button("Process Podcast Feed")
    st.sidebar.markdown("**Note**: Podcast processing can take up to 5 mins, please be patient.")

    if process_button:

        # Call the function to process the URLs and retrieve podcast guest information
        try:
            podcast_info = process_podcast_info(url)
            # Display the podcast title
            st.subheader("Episode Title")
            st.write(podcast_info['podcast_details']['episode_title'])

            # Display the podcast summary and the cover image in a side-by-side layout
            col1, col2 = st.columns([7, 3])

            with col1:
                # Display the podcast episode summary
                st.subheader("Podcast Episode Summary")
                st.write(podcast_info['podcast_summary'])

            with col2:
                st.image(podcast_info['podcast_details']['episode_image'], caption="Podcast Cover", width=300, use_column_width=True)

            # Display the podcast guest and their details in a side-by-side layout
            col3, col4 = st.columns([3, 7])

            with col3:
                st.subheader("Podcast Guest")
                st.write(podcast_info['podcast_guest']['name'])

            with col4:
                st.subheader("Podcast Guest Details")
                st.write(podcast_info["podcast_guest"]['summary'])

            # Display the five key moments
            st.subheader("Key Moments")
            key_moments = podcast_info['podcast_highlights']
            for moment in key_moments.split('\n'):
                st.markdown(
                    f"<p style='margin-bottom: 5px;'>{moment}</p>", unsafe_allow_html=True)
        except:
            st.warning('There was an issue processing this RSS feed, please try another', icon='🤖')
main()