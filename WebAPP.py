# modules and libraries:
import streamlit as st
from streamlit.hashing import _CodeHasher
import cv2
import numpy as np
import PoseModule as pm
import tempfile

#   exception handler:
try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server


# main desktop function
def main():
    pages = {  # page shifting
        "Home": page_home,
        "Experiment": page_experiment,
    }
    state = _get_state()
    st.sidebar.title("*SELF'IT*")  # heading
    st.sidebar.write("---")
    st.sidebar.title("NAVIGATION")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


# Home page
def page_home(state):
    st.title("SELF'IT")  # heading
    st.subheader('_Allow ML to assist us in being more fit._')
    st.write("---")
    st.title("Home")
    st.write("This software is designed for exercises that include repetitive movement of a certain body part.   \nIt keeps track of the number of times the practise is repeated so that user don't have to bother about it.")


# Experiment page
def page_experiment(state):
    st.sidebar.write("---")
    # Input method
    state.inputMethod = st.sidebar.selectbox("Choose the app mode",
                                             ["Show instructions", "Video experiment"])
    if state.inputMethod == "Show instructions":
        st.title("Instructions")
        instructionForImplementation(state)
        st.sidebar.success('Before experimenting give inputs as instructed')
    elif state.inputMethod == "Video experiment":
        st.title("Video Experiment")
        st.text('* Before proceeding please read the instruction')
        display_state_values(state)
        pageExperiment_Video(state)


# instruction for experiment
def instructionForImplementation(state):
    st.write("Choose three- *body landmarks* according to the exercise you want to perform:   ")
    st.image('AiTrainer/blazePose33Points.jpg' , "Pose landmarks.")

    colA, col1, col2, col3 = st.beta_columns(4)
    with colA:
        state.pose = st.text_input("Body Part: ", state.pose or "")
    with col1:
        state.LandMark1 = st.text_input("LandMark-1 ", state.LandMark1 or "")
    with col2:
        state.LandMark2 = st.text_input("LandMark-2 ", state.LandMark2 or "")
    with col3:
        state.LandMark3 = st.text_input("LandMark-3: ", state.LandMark3 or "")
    st.write('For example:    \nIf you\'re going to perform _push-ups_ :    \n')
    colA, col1, col2, col3 = st.beta_columns(4)
    with colA:
        st.text('Body Part: Left Hand')
    with col1:
        st.text('LandMark-1: 11')
    with col2:
        st.text('LandMark-2: 13')
    with col3:
        st.text('LandMark-3: 15')


# for video insertion & demo
def pageExperiment_Video(state):
    # name
    state.input = st.sidebar.text_input("Name: ", state.input or "")
    # video uploading
    state.checkbox = st.sidebar.checkbox("Upload video.", state.checkbox)

    # angle adjustment
    state.slider1 = st.sidebar.slider("Set minimum angle", 10, 360, state.slider1)
    state.slider2 = st.sidebar.slider("Set maximum angle", 50, 360, state.slider2)

    # position
    state.radio = st.sidebar.radio("Set workout position: ", ["Upper", "Lower", "Core"])
    if state.radio == "Upper":
        position_Upper(state)
    elif state.radio == "Lower":
        position_Lower(state)
    elif state.radio == "Core":
        position_Core(state)

    # for uploading video
    if state.checkbox:
        fileUploadedCode(state)

# Final value displaying
def display_state_values(state):
    st.write("Name:", state.input)
    st.write("Input method:", state.inputMethod)
    st.write("Body part:", state.pose)
    st.write("Body Position: ", state.radio)
    st.write("Uploaded Video:", state.checkbox)
    st.write("Selected workout:", state.selectbox)
    st.write("Minimum angle b/w joint:", state.slider1)
    st.write("Maximum angle b/w joint:", state.slider2)
    st.write("Score:", state.score)

    if st.button("Clear state"):
        state.clear()
    st.write("---")


# upper body exercise
def position_Upper(state):
    # exercise
    state.selectbox = st.sidebar.selectbox("Select particular exercise.", ["Pull Ups", "Biceps"])
    if state.selectbox == "Pull Ups":
        video = 'pullUps.mp4'
        if not state.inputMethod == "Web camera" and not state.checkbox:
            st.sidebar.video('AiTrainer/' + video)

        #  if video is not uploaded
        if not state.checkbox:
            workout_Code(state, video)


    elif state.selectbox == "Biceps":
        video = 'biceps.mp4'
        if not state.inputMethod == "Web camera" and not state.checkbox:
            st.sidebar.video('AiTrainer/' + video)
        #  if video is not uploaded
        if not state.checkbox:
            workout_Code(state, video)


def position_Lower(state):
    # exercise
    state.selectbox = st.sidebar.selectbox("Select particular exercise.", ["Sit Ups", "Squat"])
    if state.selectbox == 'Sit Ups':
        video = "sitUps.mp4"
        if not state.inputMethod == "Web camera" and not state.checkbox:
            st.sidebar.video('AiTrainer/' + video)
        #  if video is not uploaded
        if not state.checkbox:
            workout_Code(state, video)
    elif state.selectbox == 'Squat':
        video = "squat.mp4"
        if not state.inputMethod == "Web camera" and not state.checkbox:
            st.sidebar.video('AiTrainer/' + video)
        #  if video is not uploaded
        if not state.checkbox:
            workout_Code(state, video)


# whole body
def position_Core(state):
    # exercise
    state.selectbox = st.sidebar.selectbox("Select particular exercise.", ["Push Ups", "--"])
    if state.selectbox == "Push Ups":
        video = 'pushUp1.mp4'
        if not state.inputMethod == "Web camera" and not state.checkbox:
            st.sidebar.video('AiTrainer/' + video)
        #  if video is not uploaded
        if not state.checkbox:
            workout_Code(state, video)



def workout_Code(state, video):
    st.sidebar.text_input("Count", state.score)
    # get_plot(state)
    stframe = st.empty()

    if st.button('Start'):
        f = open('AiTrainer/' + video, 'rb')
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(f.read())
        vf = cv2.VideoCapture(tfile.name)

        detector = pm.poseDetector()
        count = 0
        dir = 0
        angle = state.slider1
        while vf.isOpened():
            success, frame = vf.read()
            # if frame is read correctly ret is True
            if not success:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            img = detector.findPose(frame, False)
            lmList = detector.findPosition(frame, False)
            if len(lmList) != 0:
                angle = detector.findAngle(img, int(state.LandMark1), int(state.LandMark2), int(state.LandMark3))
                per = np.interp(angle, (state.slider1, state.slider2), (0, 100))
                if per == 100:
                    if dir == 0:
                        count += 0.5
                        dir = 1
                if per == 0:
                    if dir == 1:
                        count += 0.5
                        dir = 0
            stframe.image(frame, channels="BGR")
            state.score = count
            # state.angle = angle
    elif st.button("Stop"):
        st.stop()


def fileUploadedCode(state):
    f = st.file_uploader("Upload file")
    if f is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(f.read())

        vf = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        st.sidebar.video(f)

        if st.button('Start'):
            detector = pm.poseDetector()
            count = 0
            dir = 0
            angle = state.slider1
            while vf.isOpened():
                success, frame = vf.read()
                # if frame is read correctly ret is True
                if not success:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                img = detector.findPose(frame, False)
                lmList = detector.findPosition(frame, False)
                if len(lmList) != 0:
                    angle = detector.findAngle(img, int(state.LandMark1), int(state.LandMark2), int(state.LandMark3))
                    per = np.interp(angle, (state.slider1, state.slider2), (0, 100))
                    if per == 100:
                        if dir == 0:
                            count += 0.5
                            dir = 1
                    if per == 0:
                        if dir == 1:
                            count += 0.5
                            dir = 0
                stframe.image(frame, channels="BGR")
                state.score = count
                # state.angle = angle
        elif st.button("Stop"):
            st.stop()

# driver code
class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()
