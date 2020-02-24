import numpy as np


class DepthMap:
    """Define class for calculate depth map"""
    def __init__(self, disparition_array, img_1_left, img_2_right):
        self.disparition_array = disparition_array
        self.img_1_left = img_1_left
        self.img_2_right = img_2_right
        self.width = img_2_right.shape[1]
        self.height = img_2_right.shape[0]
        self.f_pred = None
        self.s_pred = None

    def get_vertices(self, norm):
        vertices = np.zeros((self.disparition_array.shape[0], self.disparition_array.shape[0]))
        for i in self.disparition_array:
            for j in self.disparition_array:
                if norm == 'L1':
                    vertices[i, j] = np.abs(i - j)
                elif norm == 'L2':
                    vertices[i, j] = np.power(i - j, 2)
        return vertices

    def _get_h_for_array(self, index, disp, norm):
        if (index - disp) < 0:
            return None
        else:
            diff = self.img_1_left[:, index] - self.img_2_right[:, index - disp]
            if norm == "L1":
                return np.abs(diff)
            elif norm == "L2":
                return np.power(diff, 2)

    def _get_arr_pred_for_next_arr(self, index_next, arr_next, vertices):
        min_matrix = np.zeros((self.height, self.f_pred.shape[1]))

        for arr_previous in range(0, self.f_pred.shape[1]):
            next_values = self.f_pred[:, arr_previous] + (vertices[arr_previous, arr_next] * 50)  # alpha
            min_matrix[:, arr_previous] = next_values

        min_values = min_matrix.min(axis=1)
        min_values = np.add(min_values, self._get_h_for_array(index_next, arr_next, "L1"))
        min_args = min_matrix.argmin(axis=1)
        return min_values, min_args

    def calculate_depth_map(self, vertices):
        assert (self.img_1_left.shape == self.img_2_right.shape)

        for index in range(self.width):
            if index == 0:
                f_array = np.zeros((self.img_1_left.shape[0], 1))
                s_array = np.zeros((self.img_1_left.shape[0], 1, 1))

                for disp in self.disparition_array:
                    current_nodes = self._get_h_for_array(0, disp, "L2")
                    if current_nodes is None:
                        self.f_pred = None
                        self.s_pred = None
                        break
                    f_array = current_nodes.reshape(self.img_1_left.shape[0], 1)
                    s_array[:, disp, disp] = 0
                self.f_pred = f_array
                self.s_pred = s_array

            else:
                if self.f_pred.shape[1] == len(self.disparition_array):
                    width = len(self.disparition_array)
                else:
                    width = self.f_pred.shape[1] + 1

                f_array = np.zeros((self.img_1_left.shape[0], width))
                width_s = self.s_pred.shape[2] + 1
                s_array = np.zeros((self.img_1_left.shape[0], width, width_s))

                for w in range(width):
                    next_values, previous_ds = self._get_arr_pred_for_next_arr(index, w, vertices)

                    f_array[:, w] = next_values

                    for i, prev_d in enumerate(previous_ds):
                        s_array[i, w, :-1] = self.s_pred[i, prev_d, :]
                        s_array[i, w, -1] = w

                self.f_pred = f_array
                self.s_pred = s_array
