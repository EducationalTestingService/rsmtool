<#  
.SYNOPSIS  
    Distribute the tests in VSTS pipeline across multiple agents 
.DESCRIPTION  
    This script divides test files across multiple agents for running on Azure DevOps.
    It is adapted from the script in this repository: 
    https://github.com/PBoraMSFT/ParallelTestingSample-Python/blob/master/DistributeTests.ps1

    The distribution is basically identical to the way we do it in .travis.yaml
#>

$tests = Get-ChildItem .\tests\ -Filter "test*.py" # search for test files with specific pattern.
$totalAgents = [int]$Env:SYSTEM_TOTALJOBSINPHASE # standard VSTS variables available using parallel execution; total number of parallel jobs running
$agentNumber = [int]$Env:SYSTEM_JOBPOSITIONINPHASE  # current job position
$testCount = $tests.Count

# below conditions are used if parallel pipeline is not used. i.e. pipeline is running with single agent (no parallel configuration)
if ($totalAgents -eq 0) {
    $totalAgents = 1
}
if (!$agentNumber -or $agentNumber -eq 0) {
    $agentNumber = 1
}

Write-Host "Total agents: $totalAgents"
Write-Host "Agent number: $agentNumber"
Write-Host "Total tests: $testCount"

$testsToRun= @()

if ($agentNumber -eq 1) {
    $testsToRun = $testsToRun + "tests/test_experiment_rsmtool_1.py"
}
elseif ($agentNumber -eq 2) {
    $testsToRun = $testsToRun + "tests/test_comparer.py"
    $testsToRun = $testsToRun + "tests/test_configuration_parser.py"
    $testsToRun = $testsToRun + "tests/test_experiment_rsmtool_2.py"
    $testsToRun = $testsToRun + "tests/test_container.py"
}
elseif ($agentNumber -eq 3) {
    $testsToRun = $testsToRun + "tests/test_analyzer.py"
    $testsToRun = $testsToRun + "tests/test_experiment_rsmeval.py"
    $testsToRun = $testsToRun + "tests/test_fairness_utils.py"
    $testsToRun = $testsToRun + "tests/test_prmse_utils.py"
    $testsToRun = $testsToRun + "tests/test_test_utils.py"
    $testsToRun = $testsToRun + "tests/test_cli.py"
}
elseif ($agentNumber -eq 4) {
    $testsToRun = $testsToRun + "tests/test_experiment_rsmcompare.py"
    $testsToRun = $testsToRun + "tests/test_experiment_rsmsummarize.py"
    $testsToRun = $testsToRun + "tests/test_modeler.py"
    $testsToRun = $testsToRun + "tests/test_preprocessor.py"
    $testsToRun = $testsToRun + "tests/test_writer.py"
    $testsToRun = $testsToRun + "tests/test_experiment_rsmtool_3.py"
}
elseif ($agentNumber -eq 5) {
    $testsToRun = $testsToRun + "tests/test_experiment_rsmpredict.py"
    $testsToRun = $testsToRun + "tests/test_reader.py"
    $testsToRun = $testsToRun + "tests/test_reporter.py"
    $testsToRun = $testsToRun + "tests/test_transformer.py"
    $testsToRun = $testsToRun + "tests/test_utils.py"
    $testsToRun = $testsToRun + "tests/test_experiment_rsmtool_4.py"
}

# join all test files seperated by space. pytest runs multiple test files in following format pytest test1.py test2.py test3.py
$testFiles = $testsToRun -Join " "
Write-Host "Test files $testFiles"
# write these files into variable so that we can run them using pytest in subsequent task. 
Write-Host "##vso[task.setvariable variable=pytestfiles;]$testFiles" 
