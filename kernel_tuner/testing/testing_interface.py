import itertools
from kernel_tuner import core, util
from kernel_tuner.integration import create_results, get_objective_defaults
from kernel_tuner.interface import run_kernel, tune_kernel
from kernel_tuner.testing.mutation.mutant import HigherOrderMutant
from kernel_tuner.testing.mutation.mutation_analyzer import MutationAnalyzer
from kernel_tuner.testing.mutation.mutation_exectutor import MutationExecutor
from kernel_tuner.testing.testing_kernel import TestingKernelBuilder
from kernel_tuner.testing.mutation.mutation_operator import loadAllOperators
from kernel_tuner.testing.mutation.mutation_result import MutationResult
from kernel_tuner.testing.testing_test_case import filter_by_problem_size, verification
from kernel_tuner.testing.testing_result import TestingResult

def test_kernel(
        kernel_name,
        kernel_source,
        test_cases,
        test_params,
        grid_div_x=None,
        grid_div_y=None,
        grid_div_z=None,
        restrictions=None,
        verbose=False,
        lang=None,
        device=0,
        platform=0,
        smem_args=None,
        cmem_args=None,
        texmem_args=None,
        compiler=None,
        compiler_options=None,
        defines=None,
        log=None,
        iterations=7,
        block_size_names=None,
        quiet=False,
        strategy=None,
        strategy_options=None,
        cache=None,
        metrics=None,
        simulation_mode=False,
        observers=None,
        objective=None,
        objective_higher_is_better=None,
) -> TestingResult:

    if not isinstance(test_cases, list) or len(test_cases)==0:
        raise RuntimeError("invalid test cases")
    
    #TODO: only support single kernel_string
    kernelsource = core.KernelSource(kernel_name, kernel_source, lang, defines)
    kernel_string = kernelsource.get_kernel_string(0)
    
    # default objective if none is specified
    objective, objective_higher_is_better = get_objective_defaults(objective, objective_higher_is_better)
    
    problem_size_list, filtered_test_cases = filter_by_problem_size(test_cases)

    best_config_list = []
    tune_meta = None
    tune_data = None
    testing_result = TestingResult(test_cases)

    for idx, problem_size in enumerate(problem_size_list):
        #Retrieve the first test case for each problem size and tune it
        tune_case = filtered_test_cases[idx][0]
        #TODO: fill out all the parameters
        #TODO: check test_params is list or not before input
        try:
            tune_results, env = tune_kernel(kernel_name, kernel_source, tune_case.problem_size, tune_case.input, test_params,
                                grid_div_x, grid_div_y, grid_div_z, restrictions, tune_case.output, tune_case.atol, tune_case.verify,
                                verbose, lang, device, platform, smem_args, cmem_args, texmem_args, compiler, compiler_options,
                                defines, log, iterations, block_size_names, quiet, strategy, strategy_options, cache, metrics,
                                simulation_mode, observers, objective, objective_higher_is_better)
            tune_case.test_pass()
        except RuntimeError as e:
            error_msg = str(e)
            if  "Kernel result verification failed" in error_msg:
                tune_case.test_fail(error_msg)
                print("Test fail for turning test case")
                continue
            else: raise e
        tune_meta, tune_data = create_results(kernel_name, kernel_string, test_params, tune_case.problem_size,
                                     tune_results, env, meta = tune_meta, data = tune_data)
        best_config = util.get_best_config(tune_results, objective, objective_higher_is_better)
        best_config_list.append(best_config)

        #Verify all the test cases
        if len(filtered_test_cases[idx]) > 1:
            for test_case_idx in range(1, len(filtered_test_cases[idx])):
                #TODO: Verify all the results of test cases
                case = filtered_test_cases[idx][test_case_idx]
                #TODO: fill out all the parameters
                result = run_kernel(kernel_name, kernel_source, case.problem_size, case.input, best_config,
                                    grid_div_x, grid_div_y, grid_div_z, lang, device, platform,
                                    smem_args, cmem_args, texmem_args, compiler,compiler_options,
                                    defines, block_size_names, quiet, log)
                case = verification(result, case)
    
    return testing_result

def mut_kernel(
        kernel_name,
        kernel_source,
        test_cases,
        test_params,
        mutation_order = 1,
        mutation_analyze_only = False,
        mutation_timeout_second = 10,
        grid_div_x=None,
        grid_div_y=None,
        grid_div_z=None,
        restrictions=None,
        verbose=False,
        lang=None,
        device=0,
        platform=0,
        smem_args=None,
        cmem_args=None,
        texmem_args=None,
        compiler=None,
        compiler_options=None,
        defines=None,
        log=None,
        iterations=7,
        block_size_names=None,
        quiet=False,
        strategy=None,
        strategy_options=None,
        cache=None,
        metrics=None,
        simulation_mode=False,
        observers=None,
        objective=None,
        objective_higher_is_better=None,
):
    if not isinstance(test_cases, list) or len(test_cases)==0:
        raise RuntimeError("invalid test cases")
    
    #TODO: only support single kernel_string
    kernelsource = core.KernelSource(kernel_name, kernel_source, lang, defines)
    kernel_string = kernelsource.get_kernel_string(0)
    
    # default objective if none is specified
    objective, objective_higher_is_better = get_objective_defaults(objective, objective_higher_is_better)
    
    problem_size_list, filtered_test_cases = filter_by_problem_size(test_cases)
    best_config_list = []
    tune_meta = None
    tune_data = None
    for idx, problem_size in enumerate(problem_size_list):
        #Retrieve the first test case for each problem size and tune it
        tune_case = filtered_test_cases[idx][0]
        #TODO: fill out all the parameters
        try:
            tune_results, env = tune_kernel(kernel_name, kernel_source, tune_case.problem_size, tune_case.input, test_params,
                                grid_div_x, grid_div_y, grid_div_z, restrictions, tune_case.output, tune_case.atol, tune_case.verify,
                                verbose, lang, device, platform, smem_args, cmem_args, texmem_args, compiler, compiler_options,
                                defines, log, iterations, block_size_names, quiet, strategy, strategy_options, cache, metrics,
                                simulation_mode, observers, objective, objective_higher_is_better)
            tune_case.test_pass()
        except RuntimeError as e:
            error_msg = str(e)
            if  "Kernel result verification failed" in error_msg:
                tune_case.test_fail(error_msg)
            else:
                raise e
        tune_meta, tune_data = create_results(kernel_name, kernel_string, test_params, tune_case.problem_size,
                                     tune_results, env, meta = tune_meta, data = tune_data)
        best_config = util.get_best_config(tune_results, objective, objective_higher_is_better)
        best_config_list.append(best_config)

        #Verify all the test cases
        if len(filtered_test_cases[idx]) > 1:
            for test_case_idx in range(1, len(filtered_test_cases[idx])):
                #TODO: Verify all the results of test cases
                case = filtered_test_cases[idx][test_case_idx]
                #TODO: fill out all the parameters
                result = run_kernel(kernel_name, kernel_source, case.problem_size, case.input, best_config)
                case = verification(result, case)
    
    operators = loadAllOperators()

    analyzer = MutationAnalyzer(kernelsource, operators)
    mutants = analyzer.analyze()
    if mutation_analyze_only:
        analyze_result = MutationResult(mutants, test_cases)
        analyze_result.add_tune_data_meta(tune_meta, tune_data)
        #TODO: dump file

    # Check if high order mutation testing
    ho_mutants = []
    if mutation_order > 1:
        comb_mutants_list = itertools.combinations(mutants, mutation_order)
        ho_mutants = [HigherOrderMutant(
            id = "Combi-"+("-".join([str(mutant.id) for mutant in comb_mutants])),
            mutants = comb_mutants,
            mutation_order = mutation_order) for comb_mutants in comb_mutants_list]

    # Execute the mutants and ho_mutants (if any)
    for idx, problem_size in enumerate(problem_size_list):
        test_case_0 = filtered_test_cases[idx][0]
        builder = TestingKernelBuilder(kernel_name, kernel_string, problem_size, 
                                       test_case_0.input, test_case_0.output, best_config_list[idx]) \
                    .add_grid_div(grid_div_x, grid_div_y, grid_div_z) \
                    .add_restriction(restrictions) \
                    .enable_testing_timeout(mutation_timeout_second)
        executor = MutationExecutor(builder, mutants, filtered_test_cases[idx], ho_mutants, mutation_timeout_second)
        print("Start mutation testing using test cases [", [", ".join(str(case.id)) for case in filtered_test_cases[idx]], "], problem size", problem_size)
        executor.execute()

    return MutationResult(mutants, test_cases, ho_mutants)
