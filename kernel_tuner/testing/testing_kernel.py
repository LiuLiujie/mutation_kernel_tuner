import numpy as np
from kernel_tuner import core
from kernel_tuner.interface import Options, _kernel_options

from kernel_tuner.integration import TuneResults


class TestingKernel():
    def __init__(self, kernel_name, kernel_string, params, arguments, expected_output, kernel_options, verbose, device=0, lang=None):
        self.kernel_name = kernel_name
        self.kernel_string = kernel_string
        self.kernel_options = kernel_options
        self.kernel_params = params
        self.arguments = arguments
        self.expected_output = expected_output
        self.verbose = verbose
        self.device = device
        self.lang = lang

        kernel_source = core.KernelSource(self.kernel_name, self.kernel_string, self.lang)
        self.dev_inf = core.DeviceInterface(kernel_source, device=self.device, quiet=True)

        self.update_kernel(self.kernel_string)

        #setup GPU memory
        self.gpu_args = self.dev_inf.ready_argument_list(self.arguments)
        self.update_gpu_args(self.arguments)
    
    def update_gpu_args(self, args) -> None:
        self.arguments = args
        for i, arg in enumerate(args):
            if isinstance(args[i], np.ndarray):
                self.dev_inf.dev.memcpy_htod(self.gpu_args[i], arg)
            else:
                self.gpu_args[i] = arg

    def update_kernel(self, kernel_string) -> None:
        self.kernel_string = kernel_string
        # TODO: check if it will throw errors

        #load the kernel source code
        kernel_source = core.KernelSource(self.kernel_name, self.kernel_string, self.lang)

        #instantiate the kernel given the parameters in params
        self.kernel_instance = self.dev_inf.create_kernel_instance(kernel_source, self.kernel_options, self.kernel_params, self.verbose)

        #compile the kernel
        self.func = self.dev_inf.compile_kernel(self.kernel_instance, self.verbose)

    def __reset_gpu_result(self) -> None:
        for i, arg in enumerate(self.arguments):
            if self.expected_output[i] is not None:
                if isinstance(arg, np.ndarray):
                    self.dev_inf.dev.memcpy_htod(self.gpu_args[i], arg)
                else:
                    self.gpu_args[i] = arg
            
    def __get_gpu_result(self) -> list:
        results = []
        for i, _ in enumerate(self.expected_output):
            if isinstance(self.expected_output[i], np.ndarray):
                res = np.zeros_like(self.expected_output[i])
                self.dev_inf.memcpy_dtoh(res, self.gpu_args[i])
                results.append(res)
        return results

    def verify(self, result, expected_output, verify, atol) -> bool:
        try:
            if verify:
                return verify(expected_output, result, atol)
            else:
                return core._default_verify_function(self.kernel_instance, expected_output, result, atol, self.verbose)
        except:
            return False
            
    def execute(self) -> list:
        self.dev_inf.run_kernel(self.func, self.gpu_args, self.kernel_instance)
        result = self.__get_gpu_result()
        self.__reset_gpu_result()
        return result

class TestingKernelBuilder():
    def __init__(self, kernel_name, kernel_string, problem_size, arguments, expected_output, params):
        # Required parameters
        self.kernel_name = kernel_name
        self.kernel_string = kernel_string
        self.problem_size = problem_size
        self.arguments = arguments
        self.expected_output = expected_output
        self.params = params

        # Optional parameters
        self.grid_div_x=None
        self.grid_div_y=None
        self.grid_div_z=None
        self.restrictions=None
        self.device=0
        self.platform=0
        self.block_size_names=None
        self.verbose=True
        self.lang=None
    
    def add_grid_div(self, grid_div_x=None, grid_div_y=None, grid_div_z=None):
        self.grid_div_x = grid_div_x
        self.grid_div_y = grid_div_y
        self.grid_div_z = grid_div_z
        return self
    
    def add_restriction(self, restrictions):
        self.restrictions = restrictions
        return self

    def init_by_tune_result(results_file: TuneResults, arguments, expected_output):
        results = TuneResults(results_file)
        kernel_name = results.meta["kernel_name"]
        kernel_string = results.meta["kernel_string"]
        device_name = results.data["device_name"]
        problem_size = results.data["problem_size"]
        params = results.get_best_config(device_name, problem_size)
        #objective = results.meta["objective"]
        #objective_higher_is_better = results.meta.get("objective_higher_is_better", False)
        return TestingKernelBuilder(kernel_name, kernel_string, problem_size, arguments, expected_output, params)
    
    def build(self) -> TestingKernel:
        opts = self.__dict__
        kernel_options = Options([(k, opts[k]) for k in _kernel_options.keys() if k in opts.keys()])
        return TestingKernel(self.kernel_name, self.kernel_string, self.params, self.arguments,
                               self.expected_output, kernel_options, self.verbose, self.device, self.lang)