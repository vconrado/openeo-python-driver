#!/usr/bin/env groovy

/*
    This Jenkinsfile is used to provide snapshot builds using the VITO CI system. Travis is used to provide publicly accessible test results.
    This Jenkinsfile uses the Jenkins shared library. (ssh://git@git.vito.local:7999/biggeo/jenkinslib.git)
    Information about the pythonPipeline method can be found in pythonPipeline.groovy
*/

@Library('lib')_

pythonPipeline {
  package_name = 'openeo_driver'
  wipeout_workspace = true
  python_version = ["3.8"]
  downstream_job = 'openEO/openeo-integrationtests'
  wheel_repo = 'python-openeo'
  extras_require = 'dev'
  upload_dev_wheels = false
  pep440 = true
  custom_test_image = 'vito-docker.artifactory.vgt.vito.be/centos8-spark-py-openeo:3.2.0'
}
