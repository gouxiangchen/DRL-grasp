from VisualGrasp.divide_environment import EnvGrasp
from VisualGrasp.divide_model import CNNRotationSup224
from VisualGrasp.OrientationSupTrain import SupTrain
from itertools import count
from tensorboardX import SummaryWriter
import time
import numpy as np

env = EnvGrasp()
st = SupTrain(CNNRotationSup224)
k = 0
# st.load_model('sup_model_1d_r1999_.para')
for t in count():
    k += 1
    state, frame, pos = env.reset()
    # print(frame.shape)
    label = env.get_target_orientation()
    out = st.get_out(frame)
    # print(label, out)
    st.buffer.add((frame, label))
    if k > 32:
        st.learn()
    if t % 1000 == 999:
        st.save_model('sup_model_1d_r' + str(t) + '_.para')
