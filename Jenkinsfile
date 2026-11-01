pipeline {
  agent any

  parameters {
    choice(name: 'DATA_SOURCE', choices: ['dvc', 'mounted'],
           description: 'dvc = dvc pull from the Google Drive remote; mounted = use /datasets/train.csv mounted into the container')
    choice(name: 'CANDIDATE', choices: ['standard', 'imbalanced', 'reweighted'],
           description: 'Training strategy (candidate.strategy in params.yaml): standard = production recipe; imbalanced = Candidate A, drops non-toxic comments mentioning one identity; reweighted = Candidate B, up-weights non-toxic identity mentions')
    string(name: 'RUN_LABEL', defaultValue: '',
           description: 'Label recorded in MLflow (run name and candidate tag); empty = the CANDIDATE value')
    string(name: 'MAX_ROWS', defaultValue: '',
           description: 'Override data.max_rows for this build (empty = value in params.yaml). The gate stays fair at any size: every candidate is scored on the same identity hold-out.')
    booleanParam(name: 'DEPLOY', defaultValue: false,
                 description: 'Build the API image and roll it out to Kubernetes after promotion')
    string(name: 'IMAGE', defaultValue: 'toxicity-api',
           description: 'API image name used when DEPLOY is enabled')
  }

  // Check the repository every 5 minutes and build when there is a new commit.
  // Triggered builds use the default parameters (standard candidate, gated).
  triggers {
    pollSCM('H/5 * * * *')
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
        // API packages for tests/test_api.py; DVC is pinned here because
        // requirements-train.txt does not include it.
        sh '''
          uv venv --python 3.12 "$VENV"
          uv pip install --python "$VENV/bin/python" -r requirements-train.txt -r requirements-api.txt "dvc[gdrive]==3.67.1"
        '''
      }
    }

    stage('Fetch Data') {
      steps {
        script {
          if (params.DATA_SOURCE == 'dvc') {
            // Google Drive login saved by a local `dvc pull`, stored in Jenkins
            // as a "Secret file" credential with id gdrive-user-credentials.
            // The cache dir persists across builds, so only new data downloads.
            withCredentials([file(credentialsId: 'gdrive-user-credentials', variable: 'GDRIVE_CREDS')]) {
              sh '''
                . "$VENV/bin/activate"
                dvc cache dir --local /var/jenkins_home/dvc-cache
                # .dvc/config enables a service account (used by GitHub Actions);
                # this job signs in with a user login instead.
                dvc remote modify --local gdrive gdrive_use_service_account false
                set +x  # keep the token out of the build log
                export GDRIVE_CREDENTIALS_DATA="$(cat "$GDRIVE_CREDS")"
                set -x
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
          python -c "import yaml,sys; p=yaml.safe_load(open('params.yaml')); p['candidate']['strategy']=sys.argv[1]; yaml.safe_dump(p, open('params.yaml','w'), sort_keys=False)" "${CANDIDATE:-standard}"
          echo "candidate.strategy set to ${CANDIDATE:-standard}"
          dvc repro
        '''
      }
    }

    // After the pipeline: tests/test_data.py reads data/processed/ and
    // tests/test_api.py loads the trained artifacts.
    stage('Unit Tests') {
      steps {
        sh '''
          . "$VENV/bin/activate"
          mkdir -p reports
          python -m pytest -q tests --junitxml=reports/junit.xml
        '''
      }
      post { always { junit allowEmptyResults: true, testResults: 'reports/junit.xml' } }
    }

    stage('Log to MLflow') {
      steps {
        sh '''
          . "$VENV/bin/activate"
          python src/log_mlflow.py --run-name "${RUN_LABEL:-${CANDIDATE:-standard}}"
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
