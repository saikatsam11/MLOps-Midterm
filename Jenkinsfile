pipeline {
  agent any

  parameters {
    choice(name: 'DATA_SOURCE', choices: ['dvc', 'mounted'],
           description: 'dvc = dvc pull from the Google Drive remote; mounted = use /datasets/train.csv mounted into the container')
    string(name: 'RUN_LABEL', defaultValue: 'baseline',
           description: 'Candidate label recorded in MLflow (run name and candidate tag)')
    string(name: 'MAX_ROWS', defaultValue: '',
           description: 'Override data.max_rows for this build (empty = value in params.yaml). Keep equal to production for a fair gate.')
    booleanParam(name: 'DEPLOY', defaultValue: false,
                 description: 'Build the API image and roll it out to Kubernetes after promotion')
    string(name: 'IMAGE', defaultValue: 'toxicity-api',
           description: 'API image name used when DEPLOY is enabled')
  }

  options {
    timestamps()
    disableConcurrentBuilds()
    timeout(time: 45, unit: 'MINUTES')
  }

  environment {
    VENV = "${WORKSPACE}/.venv"
    UV_CACHE_DIR = "/var/jenkins_home/.cache/uv"
    PYTHONUNBUFFERED = '1'
    MLFLOW_DISABLE_AGENT_HINT = '1'
    // MLFLOW_TRACKING_URI comes from docker-compose (http://mlflow:5000).
  }

  stages {
    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Setup') {
      steps {
        sh '''
          uv venv --python 3.12 "$VENV"
          uv pip install --python "$VENV/bin/python" -r requirements-train.txt
        '''
      }
    }

    stage('Unit Tests') {
      steps {
        sh '''
          . "$VENV/bin/activate"
          mkdir -p reports
          pytest -q tests --junitxml=reports/junit.xml
        '''
      }
      post { always { junit allowEmptyResults: true, testResults: 'reports/junit.xml' } }
    }

    stage('Fetch Data') {
      steps {
        script {
          if (params.DATA_SOURCE == 'dvc') {
            // Service-account JSON stored in Jenkins as a "Secret file"
            // credential with id gdrive-service-account.
            withCredentials([file(credentialsId: 'gdrive-service-account', variable: 'GDRIVE_SA')]) {
              sh '''
                . "$VENV/bin/activate"
                dvc cache dir --local /var/jenkins_home/dvc-cache
                dvc remote modify --local gdrive gdrive_use_service_account true
                dvc remote modify --local gdrive gdrive_service_account_json_file_path "$GDRIVE_SA"
                dvc pull train.csv
              '''
            }
          } else {
            sh '''
              test -f /datasets/train.csv || { echo "No /datasets/train.csv mounted (see DATASET_DIR in jenkins/docker-compose.yml)"; exit 1; }
              ln -sf /datasets/train.csv train.csv
            '''
          }
        }
      }
    }

    stage('Reproduce Pipeline') {
      steps {
        sh '''
          . "$VENV/bin/activate"
          if [ -n "$MAX_ROWS" ]; then
            python -c "import yaml,sys; p=yaml.safe_load(open('params.yaml')); p['data']['max_rows']=int(sys.argv[1]); yaml.safe_dump(p, open('params.yaml','w'), sort_keys=False)" "$MAX_ROWS"
            echo "data.max_rows overridden to $MAX_ROWS"
          fi
          dvc repro
        '''
      }
    }

    stage('Log to MLflow') {
      steps {
        sh '''
          . "$VENV/bin/activate"
          python src/log_mlflow.py --run-name "$RUN_LABEL"
        '''
      }
    }

    stage('Fetch Production Baseline') {
      steps {
        sh '''
          . "$VENV/bin/activate"
          python scripts/promote.py fetch-baseline --out metrics/prod_bias.json
        '''
      }
    }

    stage('Fairness Gate') {
      steps {
        script {
          def rc = sh(returnStatus: true, script: '''
            . "$VENV/bin/activate"
            python scripts/fairness_gate.py --candidate metrics/bias.json --production metrics/prod_bias.json
          ''')
          if (rc != 0) {
            sh '''
              . "$VENV/bin/activate"
              python scripts/promote.py reject
            '''
            error('Fairness gate blocked promotion: see the FAIL rows above and metrics/gate_report.json')
          }
        }
      }
    }

    stage('Promote') {
      steps {
        sh '''
          . "$VENV/bin/activate"
          python scripts/promote.py promote
        '''
      }
    }

    stage('Build API Image') {
      when { expression { params.DEPLOY } }
      steps {
        sh '''
          test -f docker/Dockerfile.api || { echo "docker/Dockerfile.api not found"; exit 1; }
          docker build -f docker/Dockerfile.api -t "$IMAGE:$BUILD_NUMBER" .
        '''
      }
    }

    stage('Deploy to Kubernetes') {
      when { expression { params.DEPLOY } }
      steps {
        sh '''
          kubectl set image deployment/toxicity-api api="$IMAGE:$BUILD_NUMBER"
          kubectl rollout status deployment/toxicity-api --timeout=300s
        '''
      }
    }
  }

  post {
    always {
      archiveArtifacts artifacts: 'metrics/*.json, dvc.lock, params.yaml, reports/*.xml',
                       allowEmptyArchive: true, fingerprint: true
    }
    success { echo 'Build OK: candidate passed the fairness gate and is now @production.' }
    failure { echo 'Build FAILED: check which stage broke (tests / data / pipeline / fairness gate).' }
    cleanup { cleanWs() }
  }
}
