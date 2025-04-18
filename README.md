1º Criar um ambiente. Para isso usei o ficheiro conda.yaml presente nesta pasta
  Usei o comando "conda env create -f conda.yaml" para criar o ambiente e o comando "conda activate OML" para activar o ambiente
2º Correr o notebook rumos_bank_lending_prediction - MLflow.ipynb
3º Fazer o comando: mlflow ui --backend-store-uri ./mlruns e verificar as runs acedendo ao IP http://127.0.0.1:5000 no browser
4º Correr o notebook rumos_bank_lending-prediction-RandomForest_read_model.ipynb para ler o modelo
  verifiquei que as previsões do modelo escolhido para este projecto Random Forest não correspondiam à accuracy esperada e para resolver esses problemas criei os notebooks rumos_bank_lending_prediction - MLflow Random Forest.ipynb e rumos_bank_lending-prediction-RandomForest_read_model.ipynb
  Fiz algumas runs nestes notebooks a tentar solucionar este problema
5º Criar o conteiner do docker usando o comando: docker compose up -d 
  Após criar o conteiner, sempre que se quiser correr o container abrir o Docker Desktop e iniciar o container
6º Correr o notebook rumos_bank_lending_prediction - MLflow docker.ipynb para correr os modelos no docker e criar as runs
7º Correr o rumos_bank_lending-prediction-RandomForest_read_model docker.ipynb para ler o modelo escolhido no Docker, neste caso o Random Forest

Nota: Para fazer commit das runs é preciso instalar o Git LFS com o comando git lfs install
configurar o Git LFS com o comando: git lfs track "*.pkl"
e adicionar as alterações git add .gitattributes

