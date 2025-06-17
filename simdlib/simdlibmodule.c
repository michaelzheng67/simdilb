#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <arm_neon.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) < (b) ? (b) : (a))

#define ASSERT(a)                                                              \
  if (!(a))                                                                    \
    return NULL;

#define ASSERT_PY_OBJ(a)                                                       \
  if (!(a)) {                                                                  \
    free(a);                                                                   \
    return NULL;                                                               \
  }

enum c_type { TYPE_INT, TYPE_FLOAT };

struct int_or_float {
  enum c_type type;
  int i_val;
  float f_val;
};

/*
HELPER FUNCTIONS
*/

// for arbitrarily nested list
static Py_ssize_t nested_list_len(PyObject *l) {
  Py_ssize_t res = 0;

  if (PyLong_Check(l) || PyFloat_Check(l)) {
    return 1;
  } else if (PyList_Check(l)) {
    Py_ssize_t len = PyList_Size(l);
    for (Py_ssize_t i = 0; i < len; i++) {
      PyObject *item = PyList_GetItem(l, i);
      res += nested_list_len(item);
    }
  } else {
    return 0;
  }

  return res;
}

static PyObject *flatten_list(PyObject *obj, int32_t *buffer,
                              Py_ssize_t *index) {
  if (PyLong_Check(obj)) {
    buffer[(*index)++] = (int32_t)PyLong_AsLong(obj);

  } else if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    for (Py_ssize_t i = 0; i < len; i++) {
      PyObject *item = PyList_GetItem(obj, i);
      flatten_list(item, buffer, index);
    }
  }

  return NULL;
}

static PyObject *flatten_list_float(PyObject *obj, float *buffer,
                                    Py_ssize_t *index) {
  if (PyFloat_Check(obj)) {
    buffer[(*index)++] = (float)PyFloat_AsDouble(obj);
  } else if (PyLong_Check(obj)) {
    buffer[(*index)++] = (float)PyLong_AsLong(obj);
  } else if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    for (Py_ssize_t i = 0; i < len; i++) {
      PyObject *item = PyList_GetItem(obj, i);
      flatten_list_float(item, buffer, index);
    }
  }

  return NULL;
}

// true -> float list, false -> int list
// this assumes that the presence of any float element indicates that each
// element in the list should be casted to a float
static bool is_float_list(PyObject *l) {
  bool res = false; // assume int type

  if (PyFloat_Check(l)) {
    return true;
  } else if (PyList_Check(l)) {
    Py_ssize_t len = PyList_Size(l);
    for (Py_ssize_t i = 0; i < len; i++) {
      PyObject *item = PyList_GetItem(l, i);
      res |= is_float_list(item);
    }
  } else {
    return false;
  }

  return res;
}

struct int_or_float get_int_float(PyObject *val) {
  struct int_or_float ret;
  if (PyFloat_Check(val)) {
    ret.type = TYPE_FLOAT;
    ret.f_val = (float)PyFloat_AsDouble(val);
  } else if (PyLong_Check(val)) {
    ret.type = TYPE_INT;
    ret.i_val = PyLong_AsInt(val);
    ret.f_val = (float)PyLong_AsInt(val); // need this in case list has
                                          // float elements
  }

  return ret;
}

/*
REDUCTIONS
*/

static PyObject *sum_list(PyObject *self, PyObject *args) {
  PyObject *input;

  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &input))
    return NULL;

  bool is_float = is_float_list(input);
  Py_ssize_t len = nested_list_len(input);

  if (is_float) {
    float *data = malloc(sizeof(float) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list_float(input, data, &idx);

    float32x4_t acc = vdupq_n_f32(0);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      float32x4_t v = vld1q_f32(&data[i]);
      acc = vaddq_f32(acc, v);
    }

    float sum = vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1) +
                vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3);

    for (; i < len; i++) {
      sum += data[i];
    }

    free(data);
    return PyFloat_FromDouble((double)sum);

  } else {
    int32_t *data = malloc(sizeof(int32_t) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list(input, data, &idx);

    int32x4_t acc = vdupq_n_s32(0);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      int32x4_t v = vld1q_s32(&data[i]);
      acc = vaddq_s32(acc, v);
    }

    int32_t sum = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1) +
                  vgetq_lane_s32(acc, 2) + vgetq_lane_s32(acc, 3);

    for (; i < len; i++) {
      sum += data[i];
    }

    free(data);
    return PyLong_FromLong(sum);
  }
}

static PyObject *multiply_list(PyObject *self, PyObject *args) {
  PyObject *input;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &input))
    return NULL;

  bool is_float = is_float_list(input);
  Py_ssize_t len = nested_list_len(input);

  if (is_float) {
    float *data = malloc(sizeof(float) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list_float(input, data, &idx);

    float32x4_t acc = vdupq_n_f32(1);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      float32x4_t v = vld1q_f32(&data[i]);
      acc = vmulq_f32(acc, v);
    }

    float product = vgetq_lane_f32(acc, 0) * vgetq_lane_f32(acc, 1) *
                    vgetq_lane_f32(acc, 2) * vgetq_lane_f32(acc, 3);

    for (; i < len; i++) {
      product *= data[i];
    }

    free(data);
    return PyFloat_FromDouble((double)product);

  } else {
    int32_t *data = malloc(sizeof(int32_t) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list(input, data, &idx);

    int32x4_t acc = vdupq_n_s32(1);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      int32x4_t v = vld1q_s32(&data[i]);
      acc = vmulq_s32(acc, v);
    }

    int32_t product = vgetq_lane_s32(acc, 0) * vgetq_lane_s32(acc, 1) *
                      vgetq_lane_s32(acc, 2) * vgetq_lane_s32(acc, 3);

    for (; i < len; i++) {
      product *= data[i];
    }

    free(data);
    return PyLong_FromLong(product);
  }
}

static PyObject *min_list(PyObject *self, PyObject *args) {
  PyObject *input;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &input))
    return NULL;

  bool is_float = is_float_list(input);
  Py_ssize_t len = nested_list_len(input);

  if (is_float) {
    float *data = malloc(sizeof(float) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list_float(input, data, &idx);

    float32x4_t acc = vdupq_n_f32(FLT_MAX);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      float32x4_t v = vld1q_f32(&data[i]);
      acc = vminq_f32(acc, v);
    }

    float min_val = vgetq_lane_f32(acc, 0);
    min_val = MIN(min_val, vgetq_lane_f32(acc, 1));
    min_val = MIN(min_val, vgetq_lane_f32(acc, 2));
    min_val = MIN(min_val, vgetq_lane_f32(acc, 3));

    for (; i < len; i++) {
      min_val = MIN(min_val, data[i]);
    }

    free(data);
    return PyFloat_FromDouble((double)min_val);

  } else {
    int32_t *data = malloc(sizeof(int32_t) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list(input, data, &idx);

    int32x4_t acc = vdupq_n_s32(INT32_MAX);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      int32x4_t v = vld1q_s32(&data[i]);
      acc = vminq_s32(acc, v);
    }

    int32_t min_val = vgetq_lane_s32(acc, 0);
    min_val = MIN(min_val, vgetq_lane_s32(acc, 1));
    min_val = MIN(min_val, vgetq_lane_s32(acc, 2));
    min_val = MIN(min_val, vgetq_lane_s32(acc, 3));

    for (; i < len; i++) {
      min_val = MIN(min_val, data[i]);
    }

    free(data);
    return PyLong_FromLong(min_val);
  }
}

static PyObject *max_list(PyObject *self, PyObject *args) {
  PyObject *input;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &input))
    return NULL;

  bool is_float = is_float_list(input);
  Py_ssize_t len = nested_list_len(input);

  if (is_float) {
    float *data = malloc(sizeof(float) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list_float(input, data, &idx);

    float32x4_t acc = vdupq_n_f32(-FLT_MAX);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      float32x4_t v = vld1q_f32(&data[i]);
      acc = vmaxq_f32(acc, v);
    }

    float max_val = vgetq_lane_f32(acc, 0);
    max_val = MAX(max_val, vgetq_lane_f32(acc, 1));
    max_val = MAX(max_val, vgetq_lane_f32(acc, 2));
    max_val = MAX(max_val, vgetq_lane_f32(acc, 3));

    for (; i < len; i++) {
      max_val = MAX(max_val, data[i]);
    }

    free(data);
    return PyFloat_FromDouble((double)max_val);

  } else {
    int32_t *data = malloc(sizeof(int32_t) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list(input, data, &idx);

    int32x4_t acc = vdupq_n_s32(INT32_MIN);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      int32x4_t v = vld1q_s32(&data[i]);
      acc = vmaxq_s32(acc, v);
    }

    int32_t max_val = vgetq_lane_s32(acc, 0);
    max_val = MAX(max_val, vgetq_lane_s32(acc, 1));
    max_val = MAX(max_val, vgetq_lane_s32(acc, 2));
    max_val = MAX(max_val, vgetq_lane_s32(acc, 3));

    for (; i < len; i++) {
      max_val = MAX(max_val, data[i]);
    }

    free(data);
    return PyLong_FromLong(max_val);
  }
}

// 1 if any element in the list is true, 0 otherwise
static PyObject *any_list(PyObject *self, PyObject *args) {
  PyObject *input;

  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &input))
    return NULL;

  bool is_float = is_float_list(input);
  Py_ssize_t len = nested_list_len(input);

  if (is_float) {
    float *data = malloc(sizeof(float) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list_float(input, data, &idx);

    float32x4_t acc = vdupq_n_f32(0);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      float32x4_t v = vld1q_f32(&data[i]);
      acc = vaddq_f32(acc, v);
    }

    float sum = vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1) +
                vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3);

    for (; i < len; i++) {
      sum += data[i];
    }

    int32_t res = sum > 0;

    free(data);
    return PyBool_FromLong(res);

  } else {
    int32_t *data = malloc(sizeof(int32_t) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list(input, data, &idx);

    int32x4_t acc = vdupq_n_s32(0);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      int32x4_t v = vld1q_s32(&data[i]);
      acc = vaddq_s32(acc, v);
    }

    int32_t sum = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1) +
                  vgetq_lane_s32(acc, 2) + vgetq_lane_s32(acc, 3);

    for (; i < len; i++) {
      sum += data[i];
    }

    sum = sum > 0;

    free(data);
    return PyBool_FromLong(sum);
  }
}

// 1 if all elements in the list is true, 0 otherwise
static PyObject *all_list(PyObject *self, PyObject *args) {
  PyObject *input;

  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &input))
    return NULL;

  bool is_float = is_float_list(input);
  Py_ssize_t len = nested_list_len(input);

  if (is_float) {
    float *data = malloc(sizeof(float) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list_float(input, data, &idx);

    float32x4_t acc = vdupq_n_f32(0);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      float32x4_t v = vld1q_f32(&data[i]);
      acc = vaddq_f32(acc, v);
    }

    float sum = vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1) +
                vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3);

    for (; i < len; i++) {
      sum += data[i];
    }

    int32_t res = sum == (float)len;

    free(data);
    return PyBool_FromLong(res);

  } else {
    int32_t *data = malloc(sizeof(int32_t) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list(input, data, &idx);

    int32x4_t acc = vdupq_n_s32(0);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      int32x4_t v = vld1q_s32(&data[i]);
      acc = vaddq_s32(acc, v);
    }

    int32_t sum = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1) +
                  vgetq_lane_s32(acc, 2) + vgetq_lane_s32(acc, 3);

    for (; i < len; i++) {
      sum += data[i];
    }

    sum = sum == len;

    free(data);
    return PyBool_FromLong(sum);
  }
}

/*
MAPPINGS
*/

static PyObject *add_each_list(PyObject *self, PyObject *args) {
  PyObject *input;
  PyObject *val;

  if (!PyArg_ParseTuple(args, "O!O", &PyList_Type, &input, &val))
    return NULL;

  struct int_or_float val_struct = get_int_float(val);
  bool is_float = is_float_list(input);
  Py_ssize_t len = nested_list_len(input);

  if (is_float || val_struct.type == TYPE_FLOAT) {
    float *data = malloc(sizeof(float) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list_float(input, data, &idx);

    PyObject *result = PyList_New(len);
    ASSERT_PY_OBJ(result);

    float32x4_t val_vec = vdupq_n_f32(val_struct.f_val);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      float32x4_t v = vld1q_f32(&data[i]);
      float32x4_t res = vaddq_f32(v, val_vec);

      PyList_SetItem(result, i, PyFloat_FromDouble(vgetq_lane_f32(res, 0)));
      PyList_SetItem(result, i + 1, PyFloat_FromDouble(vgetq_lane_f32(res, 1)));
      PyList_SetItem(result, i + 2, PyFloat_FromDouble(vgetq_lane_f32(res, 2)));
      PyList_SetItem(result, i + 3, PyFloat_FromDouble(vgetq_lane_f32(res, 3)));
    }

    for (; i < len; i++) {
      PyList_SetItem(result, i, PyFloat_FromDouble(data[i] + val_struct.f_val));
    }

    free(data);
    return result;

  } else {
    int32_t *data = malloc(sizeof(int32_t) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list(input, data, &idx);

    PyObject *result = PyList_New(len);
    ASSERT_PY_OBJ(result);

    int32x4_t val_vec = vdupq_n_s32(val_struct.i_val);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      int32x4_t v = vld1q_s32(&data[i]);
      int32x4_t res = vaddq_s32(v, val_vec);

      PyList_SetItem(result, i, PyLong_FromLong(vgetq_lane_s32(res, 0)));
      PyList_SetItem(result, i + 1, PyLong_FromLong(vgetq_lane_s32(res, 1)));
      PyList_SetItem(result, i + 2, PyLong_FromLong(vgetq_lane_s32(res, 2)));
      PyList_SetItem(result, i + 3, PyLong_FromLong(vgetq_lane_s32(res, 3)));
    }

    for (; i < len; i++) {
      PyList_SetItem(result, i, PyLong_FromLong(data[i] + val_struct.i_val));
    }

    free(data);
    return result;
  }
}

static PyObject *subtract_each_list(PyObject *self, PyObject *args) {
  PyObject *input;
  PyObject *val;

  if (!PyArg_ParseTuple(args, "O!O", &PyList_Type, &input, &val))
    return NULL;

  struct int_or_float val_struct = get_int_float(val);
  bool is_float = is_float_list(input);
  Py_ssize_t len = nested_list_len(input);

  if (is_float || val_struct.type == TYPE_FLOAT) {
    float *data = malloc(sizeof(float) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list_float(input, data, &idx);

    PyObject *result = PyList_New(len);
    ASSERT_PY_OBJ(result);

    float32x4_t val_vec = vdupq_n_f32(val_struct.f_val);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      float32x4_t v = vld1q_f32(&data[i]);
      float32x4_t res = vsubq_f32(v, val_vec);

      PyList_SetItem(result, i, PyFloat_FromDouble(vgetq_lane_f32(res, 0)));
      PyList_SetItem(result, i + 1, PyFloat_FromDouble(vgetq_lane_f32(res, 1)));
      PyList_SetItem(result, i + 2, PyFloat_FromDouble(vgetq_lane_f32(res, 2)));
      PyList_SetItem(result, i + 3, PyFloat_FromDouble(vgetq_lane_f32(res, 3)));
    }

    for (; i < len; i++) {
      PyList_SetItem(result, i, PyFloat_FromDouble(data[i] - val_struct.f_val));
    }

    free(data);
    return result;
  } else {

    int32_t *data = malloc(sizeof(int32_t) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list_float(input, data, &idx);

    PyObject *result = PyList_New(len);
    ASSERT_PY_OBJ(result);

    int32x4_t val_vec = vdupq_n_s32(val_struct.i_val);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      int32x4_t v = vld1q_s32(&data[i]);
      int32x4_t res = vsubq_s32(v, val_vec);

      PyList_SetItem(result, i, PyLong_FromLong(vgetq_lane_s32(res, 0)));
      PyList_SetItem(result, i + 1, PyLong_FromLong(vgetq_lane_s32(res, 1)));
      PyList_SetItem(result, i + 2, PyLong_FromLong(vgetq_lane_s32(res, 2)));
      PyList_SetItem(result, i + 3, PyLong_FromLong(vgetq_lane_s32(res, 3)));
    }

    for (; i < len; i++) {
      PyList_SetItem(result, i, PyLong_FromLong(data[i] - val_struct.i_val));
    }

    free(data);
    return result;
  }
}

static PyObject *multiply_each_list(PyObject *self, PyObject *args) {
  PyObject *input;
  PyObject *val;

  if (!PyArg_ParseTuple(args, "O!O", &PyList_Type, &input, &val))
    return NULL;

  struct int_or_float val_struct = get_int_float(val);
  bool is_float = is_float_list(input);
  Py_ssize_t len = nested_list_len(input);

  if (is_float || val_struct.type == TYPE_FLOAT) {
    float *data = malloc(sizeof(float) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list_float(input, data, &idx);

    PyObject *result = PyList_New(len);
    ASSERT_PY_OBJ(result);

    float32x4_t val_vec = vdupq_n_f32(val_struct.f_val);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      float32x4_t v = vld1q_f32(&data[i]);
      float32x4_t res = vmulq_f32(v, val_vec);

      PyList_SetItem(result, i, PyFloat_FromDouble(vgetq_lane_f32(res, 0)));
      PyList_SetItem(result, i + 1, PyFloat_FromDouble(vgetq_lane_f32(res, 1)));
      PyList_SetItem(result, i + 2, PyFloat_FromDouble(vgetq_lane_f32(res, 2)));
      PyList_SetItem(result, i + 3, PyFloat_FromDouble(vgetq_lane_f32(res, 3)));
    }

    for (; i < len; i++) {
      PyList_SetItem(result, i, PyFloat_FromDouble(data[i] * val_struct.f_val));
    }

    free(data);
    return result;

  } else {

    int32_t *data = malloc(sizeof(int32_t) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list(input, data, &idx);

    PyObject *result = PyList_New(len);
    ASSERT_PY_OBJ(result);

    int32x4_t val_vec = vdupq_n_s32(val_struct.i_val);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      int32x4_t v = vld1q_s32(&data[i]);
      int32x4_t res = vmulq_s32(v, val_vec);

      PyList_SetItem(result, i, PyLong_FromLong(vgetq_lane_s32(res, 0)));
      PyList_SetItem(result, i + 1, PyLong_FromLong(vgetq_lane_s32(res, 1)));
      PyList_SetItem(result, i + 2, PyLong_FromLong(vgetq_lane_s32(res, 2)));
      PyList_SetItem(result, i + 3, PyLong_FromLong(vgetq_lane_s32(res, 3)));
    }

    for (; i < len; i++) {
      PyList_SetItem(result, i, PyLong_FromLong(data[i] * val_struct.i_val));
    }

    free(data);
    return result;
  }
}

static PyObject *square_each_list(PyObject *self, PyObject *args) {
  PyObject *input;

  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &input))
    return NULL;

  bool is_float = is_float_list(input);
  Py_ssize_t len = nested_list_len(input);

  if (is_float) {
    float *data = malloc(sizeof(float) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list_float(input, data, &idx);

    PyObject *result = PyList_New(len);
    ASSERT_PY_OBJ(result);

    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      float32x4_t v = vld1q_f32(&data[i]);
      float32x4_t res = vmulq_f32(v, v);

      PyList_SetItem(result, i, PyFloat_FromDouble(vgetq_lane_f32(res, 0)));
      PyList_SetItem(result, i + 1, PyFloat_FromDouble(vgetq_lane_f32(res, 1)));
      PyList_SetItem(result, i + 2, PyFloat_FromDouble(vgetq_lane_f32(res, 2)));
      PyList_SetItem(result, i + 3, PyFloat_FromDouble(vgetq_lane_f32(res, 3)));
    }

    for (; i < len; i++) {
      PyList_SetItem(result, i, PyFloat_FromDouble(data[i] * data[i]));
    }

    free(data);
    return result;

  } else {
    int32_t *data = malloc(sizeof(int32_t) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list(input, data, &idx);

    PyObject *result = PyList_New(len);
    ASSERT_PY_OBJ(result);

    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      int32x4_t v = vld1q_s32(&data[i]);
      int32x4_t res = vmulq_s32(v, v);

      PyList_SetItem(result, i, PyLong_FromLong(vgetq_lane_s32(res, 0)));
      PyList_SetItem(result, i + 1, PyLong_FromLong(vgetq_lane_s32(res, 1)));
      PyList_SetItem(result, i + 2, PyLong_FromLong(vgetq_lane_s32(res, 2)));
      PyList_SetItem(result, i + 3, PyLong_FromLong(vgetq_lane_s32(res, 3)));
    }

    for (; i < len; i++) {
      PyList_SetItem(result, i, PyLong_FromLong(data[i] * data[i]));
    }

    free(data);
    return result;
  }
}

static PyObject *subtract_and_square_each_list(PyObject *self, PyObject *args) {
  PyObject *input;
  float val;

  if (!PyArg_ParseTuple(args, "O!f", &PyList_Type, &input, &val))
    return NULL;

  bool is_float = is_float_list(input);
  Py_ssize_t len = nested_list_len(input);

  if (is_float) {
    float *data = malloc(sizeof(float) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list_float(input, data, &idx);

    PyObject *result = PyList_New(len);
    ASSERT_PY_OBJ(result);

    float32x4_t val_vec = vdupq_n_f32(val);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      float32x4_t v = vld1q_f32(&data[i]);
      float32x4_t res = vsubq_f32(v, val_vec);
      res = vmulq_f32(v, v);

      PyList_SetItem(result, i, PyFloat_FromDouble(vgetq_lane_f32(res, 0)));
      PyList_SetItem(result, i + 1, PyFloat_FromDouble(vgetq_lane_f32(res, 1)));
      PyList_SetItem(result, i + 2, PyFloat_FromDouble(vgetq_lane_f32(res, 2)));
      PyList_SetItem(result, i + 3, PyFloat_FromDouble(vgetq_lane_f32(res, 3)));
    }

    for (; i < len; i++) {
      PyList_SetItem(result, i,
                     PyFloat_FromDouble(powf((data[i] - val), 2.0f)));
    }

    free(data);
    return result;

  } else {

    int32_t *data = malloc(sizeof(int32_t) * len);
    ASSERT(data && len > 0)

    Py_ssize_t idx = 0;
    flatten_list(input, data, &idx);

    PyObject *result = PyList_New(len);
    ASSERT_PY_OBJ(result);

    int32x4_t val_vec = vdupq_n_s32(val);
    Py_ssize_t i = 0;
    for (; i + 4 <= len; i += 4) {
      int32x4_t v = vld1q_s32(&data[i]);
      int32x4_t res = vsubq_s32(v, val_vec);
      res = vmulq_f32(v, v);

      PyList_SetItem(result, i, PyLong_FromLong(vgetq_lane_s32(res, 0)));
      PyList_SetItem(result, i + 1, PyLong_FromLong(vgetq_lane_s32(res, 1)));
      PyList_SetItem(result, i + 2, PyLong_FromLong(vgetq_lane_s32(res, 2)));
      PyList_SetItem(result, i + 3, PyLong_FromLong(vgetq_lane_s32(res, 3)));
    }

    for (; i < len; i++) {
      PyList_SetItem(result, i, PyLong_FromLong(pow((data[i] - val), 2)));
    }

    free(data);
    return result;
  }
}

static PyMethodDef SimdlibMethods[] = {
    {"sum_list", sum_list, METH_VARARGS,
     "Sum a list of elements using NEON SIMD"},
    {"multiply_list", multiply_list, METH_VARARGS,
     "Multiply a list of elements using NEON SIMD"},
    {"min_list", min_list, METH_VARARGS,
     "Find min from list of integers using NEON SIMD"},
    {"max_list", max_list, METH_VARARGS,
     "Find max from list of integers using NEON SIMD"},
    {"any_list", any_list, METH_VARARGS, "Find if any element is truthy"},
    {"all_list", all_list, METH_VARARGS, "Find if all elements are truthy"},
    {"add_each_list", add_each_list, METH_VARARGS,
     "Add each element with given value"},
    {"subtract_each_list", subtract_each_list, METH_VARARGS,
     "Subtract each element with given value"},
    {"multiply_each_list", multiply_each_list, METH_VARARGS,
     "Multiply each element with given value"},
    {"square_each_list", square_each_list, METH_VARARGS,
     "Square each element (n^2)"},
    {"subtract_and_square_each_list", subtract_and_square_each_list,
     METH_VARARGS,
     "Subtract and Square each element (niche op for financial calculations)"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef simdlibmodule = {
    PyModuleDef_HEAD_INIT, "simdlib_module", NULL, -1, SimdlibMethods};

PyMODINIT_FUNC PyInit_simdlib_module(void) {
  return PyModule_Create(&simdlibmodule);
}
