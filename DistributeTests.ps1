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
    $testsToRun = $testsToRun + "test_experiment_rsmtool_1"
}
elseif ($agentNumber -eq 2) {
    $testsToRun = $testsToRun + "test_comparer"
    $testsToRun = $testsToRun + "test_configuration_parser"
    $testsToRun = $testsToRun + "test_experiment_rsmtool_2"
    $testsToRun = $testsToRun + "test_container"
}
elseif ($agentNumber -eq 3) {
    $testsToRun = $testsToRun + "test_analyzer"
    $testsToRun = $testsToRun + "test_experiment_rsmeval"
    $testsToRun = $testsToRun + "test_fairness_utils"
    $testsToRun = $testsToRun + "test_utils_prmse"
    $testsToRun = $testsToRun + "test_test_utils"
    $testsToRun = $testsToRun + "test_cli"
}
elseif ($agentNumber -eq 4) {
    $testsToRun = $testsToRun + "test_experiment_rsmcompare"
    $testsToRun = $testsToRun + "test_experiment_rsmsummarize"
    $testsToRun = $testsToRun + "test_modeler"
    $testsToRun = $testsToRun + "test_preprocessor"
    $testsToRun = $testsToRun + "test_writer"
    $testsToRun = $testsToRun + "test_experiment_rsmtool_3"
}
elseif ($agentNumber -eq 5) {
    $testsToRun = $testsToRun + "test_experiment_rsmpredict"
    $testsToRun = $testsToRun + "test_reader"
    $testsToRun = $testsToRun + "test_reporter"
    $testsToRun = $testsToRun + "test_transformer"
    $testsToRun = $testsToRun + "test_utils"
    $testsToRun = $testsToRun + "test_experiment_rsmtool_4"
}
elseif ($agentNumber -eq 6) {
    $testsToRun = $testsToRun + "test_experiment_rsmxval"
    $testsToRun = $testsToRun + "test_experiment_rsmexplain"
    $testsToRun = $testsToRun + "test_explanation_utils"
}

# join all test files seperated by space. pytest runs multiple test files in following format pytest test1.py test2.py test3.py
$testFiles = $testsToRun -Join " "
Write-Host "Test files $testFiles"
# write these files into variable so that we can run them using pytest in subsequent task.
Write-Host "##vso[task.setvariable variable=pytestfiles;]$testFiles"
