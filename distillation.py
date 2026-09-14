import os
import keras
#import tensorflow
from keras import layers
#from keras import ops
#from keras.models import load_model # type: ignore
import numpy as np
import pickle
import shap
# from keras.utils import plot_model
# view the current working directory
# os.getcwd()

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(
            (teacher_pred / self.temperature),
            (y_pred / self.temperature),
        ) * (self.temperature**2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, x):
        return self.student(x)

# Prepare the train and test dataset.
with open('CNNX.pkl', 'rb') as file:
    XX = pickle.load(file)
with open('CNNY.pkl', 'rb') as file:
    YY = pickle.load(file)
with open('CNNObsp.pkl', 'rb') as file:
    Obs_p = pickle.load(file)
with open('CNNXtest.pkl', 'rb') as file:
    X_test = pickle.load(file)
with open('CNNYtest.pkl', 'rb') as file:
    Y_test = pickle.load(file)
# # Prepare the train and test dataset.
# with open('C:/Users/86182/OneDrive/桌面/DC-MA-GLM/文献/廖军/DC-GLM/simulation/real_data/CNNX.pkl', 'rb') as file:
#     XX = pickle.load(file)
# with open('C:/Users/86182/OneDrive/桌面/DC-MA-GLM/文献/廖军/DC-GLM/simulation/real_data/CNNY.pkl', 'rb') as file:
#     YY = pickle.load(file)
# with open('C:/Users/86182/OneDrive/桌面/DC-MA-GLM/文献/廖军/DC-GLM/simulation/real_data/CNNObsp.pkl', 'rb') as file:
#     Obs_p = pickle.load(file)
# with open('C:/Users/86182/OneDrive/桌面/DC-MA-GLM/文献/廖军/DC-GLM/simulation/real_data/CNNXtest.pkl', 'rb') as file:
#     X_test = pickle.load(file)
# with open('C:/Users/86182/OneDrive/桌面/DC-MA-GLM/文献/廖军/DC-GLM/simulation/real_data/CNNYtest.pkl', 'rb') as file:
#     Y_test = pickle.load(file)

epochm=np.array([50,200])#50,200
batch_sizem=np.array([128,32])
X_test=np.array(X_test)
X_test=np.expand_dims(X_test,-1)
Y_test=np.array(Y_test)
# Create matrices to save loss at each site for each model
M,nm,xdim,ydim=np.shape(XX)
Kfold=len(Obs_p[0])
mu_mat=np.zeros((2,M,nm,M)) # 2 CNN model, site model, sample size, predicted sample location
mu_test=np.zeros((2,M,np.shape(Y_test)[0])) 
shapley_cnn=np.zeros((2,M,xdim,ydim))
XX=np.array(XX)
XX=np.expand_dims(XX,-1)
YY=np.array(YY)
Obs_p1=np.zeros((M,int(nm/Kfold),Kfold))
for j in range(M):
    Obs_p_m=Obs_p[j]
    all_values = [value for values_list in Obs_p_m.values() for value in values_list]
    Obs_p_m=np.reshape(all_values,(-1,Kfold),"F")-1
    Obs_p1[j,:,:]=Obs_p_m
Obs_p=Obs_p1.astype(int)

for j in range(M):
    XX_m=XX[j]
    YY_m=YY[j]
    Obs_p_m=Obs_p[j]
    mu_cv=np.zeros((nm,1))

    #######################################
    ### cnn_1
    #######################################
    # Create the teacher
    teacher = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation = 'relu', padding="same"),
        #layers.LeakyReLU(negative_slope=0.2),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation = 'relu', padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128,activation="relu"),
        layers.Dense(64,activation="relu"),
        layers.Dense(1,activation="sigmoid"),
    ],
    name="teacher",
    )
    teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()],
    )
    # Train and evaluate teacher on data.
    teacher.fit(XX_m, YY_m, epochs=epochm[0], batch_size=batch_sizem[0],verbose=0)
    # loss, accuracy=teacher.evaluate(X_test, Y_test)

    student = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(2, (3, 3), activation = 'relu', padding="same"),
        #layers.LeakyReLU(negative_slope=0.2),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(4, (3, 3), activation = 'relu', padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(1,activation="sigmoid"),
    ],
    name="student",
    )

    # # comment simultaneously: ctrl+/
    # student.compile(
    #     optimizer=keras.optimizers.Adam(),
    #     loss=keras.losses.BinaryCrossentropy(),
    #     metrics=[keras.metrics.BinaryAccuracy()],
    # )
    # # Train and evaluate teacher on data.
    # student.fit(XX_m, YY_m, epochs=100 ,batch_size=32)
    # loss1, accuracy1=student.evaluate(X_test, Y_test)

    # student = keras.Sequential(
    # [
    #     keras.Input(shape=(28, 28, 1)),
    #     layers.Conv2D(2, (3, 3), activation = 'relu', padding="same"),
    #     #layers.LeakyReLU(negative_slope=0.2),
    #     layers.MaxPooling2D(pool_size=(2, 2)),
    #     layers.Conv2D(4, (3, 3), activation = 'relu', padding="same"),
    #     layers.MaxPooling2D(pool_size=(2, 2)),
    #     layers.Flatten(),
    #     layers.Dense(1,activation="sigmoid"),
    # ],
    # name="student",
    # )
    # Initialize and compile distiller
    distiller = Distiller(student=student, teacher=teacher)
    #distiller.build(input_shape=(None, 28, 28, 1))
    distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.BinaryAccuracy()],
    student_loss_fn=keras.losses.BinaryCrossentropy(),
    #distillation_loss_fn=keras.losses.KLDivergence(),
    distillation_loss_fn=keras.losses.MeanSquaredError(),
    alpha=0.8,
    temperature=1,
    )
    
    # Distill teacher to student
    distiller.fit(XX_m, YY_m, epochs=epochm[1], batch_size=batch_sizem[1],verbose=0)
    # lossno,loss2,accuracyno,accuracy2=distiller.evaluate(X_test,Y_test)
    for l in range(M):
        if l!=j:
            mu_mat[0,j,:,l]=distiller.predict(XX[l])[:,0]
    mu_test[0,j,:]=distiller.predict(X_test)[:,0]

    e=shap.DeepExplainer(distiller.student,XX_m)
    evalue=np.array(e.shap_values(XX_m[0:100]))
    shapley_cnn[0,j,:,:]=np.mean(evalue.reshape(100,28,28,1),axis=0)[:,:,0]

    for k in range(Kfold):
        # Create the teacher
        teacher = keras.Sequential(
        [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation = 'relu', padding="same"),
        #layers.LeakyReLU(negative_slope=0.2),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation = 'relu', padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128,activation="relu"),
        layers.Dense(64,activation="relu"),
        layers.Dense(1,activation="sigmoid"),
        ],
        name="teacher",
        )
        teacher.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
        )
        # Train and evaluate teacher on data.
        teacher.fit(np.delete(XX_m,Obs_p_m[:,k],0), np.delete(YY_m,Obs_p_m[:,k],0), epochs=epochm[0], batch_size=batch_sizem[0],verbose=0)

        student = keras.Sequential(
        [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(2, (3, 3), activation = 'relu', padding="same"),
        #layers.LeakyReLU(negative_slope=0.2),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(4, (3, 3), activation = 'relu', padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(1,activation="sigmoid"),
        ],
        name="student",
        )

        # comment simultaneously: ctrl+/
        # student.compile(
        #     optimizer=keras.optimizers.Adam(),
        #     loss=keras.losses.BinaryCrossentropy(),
        #     metrics=[keras.metrics.BinaryAccuracy()],
        # )
        # # Train and evaluate teacher on data.
        # student.fit(XX_m, YY_m, epochs=100 ,batch_size=32)
        # loss1, accuracy1=student.evaluate(X_test, Y_test)

        # Initialize and compile distiller
        distiller = Distiller(student=student, teacher=teacher)
        #distiller.build(input_shape=(None, 28, 28, 1))
        distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.BinaryAccuracy()],
        student_loss_fn=keras.losses.BinaryCrossentropy(),
        distillation_loss_fn=keras.losses.MeanSquaredError(),
        alpha=0.8,
        temperature=1,
        )

        # Distill teacher to student
        distiller.fit(np.delete(XX_m,Obs_p_m[:,k],0), np.delete(YY_m,Obs_p_m[:,k],0), epochs=epochm[1], batch_size=batch_sizem[1],verbose=0)
        mu_cv[Obs_p_m[:,k],0]=distiller.predict(XX_m[Obs_p_m[:,k]])[:,0]
    
    mu_mat[0,j,:,j]=mu_cv[:,0]

    #######################################
    ### cnn_2
    #######################################
    # Create the teacher
    teacher = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(128, (3, 3), activation = 'relu', padding="same"),
        #layers.LeakyReLU(negative_slope=0.2),
        layers.MaxPooling2D(pool_size=(4, 4)),
        layers.Flatten(),
        layers.Dense(128,activation="relu"),
        layers.Dense(1,activation="sigmoid"),
    ],
    name="teacher",
    )
    teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()],
    )
    # Train and evaluate teacher on data.
    teacher.fit(XX_m, YY_m, epochs=epochm[0], batch_size=batch_sizem[0],verbose=0)
    # teacher.evaluate(X_test,Y_test)

    student = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(5, (3, 3), activation = 'relu', padding="same"),
        #layers.LeakyReLU(negative_slope=0.2),
        layers.MaxPooling2D(pool_size=(4, 4)),
        layers.Flatten(),
        layers.Dense(1,activation="sigmoid"),
    ],
    name="student",
    )

    # # comment simultaneously: ctrl+/
    # student.compile(
    #     optimizer=keras.optimizers.Adam(),
    #     loss=keras.losses.BinaryCrossentropy(),
    #     metrics=[keras.metrics.BinaryAccuracy()],
    # )
    # # Train and evaluate teacher on data.
    # student.fit(XX_m, YY_m, epochs=100 ,batch_size=32)
    # loss1, accuracy1=student.evaluate(X_test, Y_test)

    # Initialize and compile distiller
    distiller = Distiller(student=student, teacher=teacher)
    #distiller.build(input_shape=(None, 28, 28, 1))
    distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.BinaryAccuracy()],
    student_loss_fn=keras.losses.BinaryCrossentropy(),
    distillation_loss_fn=keras.losses.MeanSquaredError(),
    alpha=0.8,
    temperature=1,
    )

    # Distill teacher to student
    distiller.fit(XX_m, YY_m, epochs=epochm[1], batch_size=batch_sizem[1],verbose=0)
    # distiller.evaluate(X_test,Y_test)
    for l in range(M):
        if l!=j:
            mu_mat[1,j,:,l]=distiller.predict(XX[l])[:,0]
    mu_test[1,j,:]=distiller.predict(X_test)[:,0]

    e=shap.DeepExplainer(distiller.student,XX_m)
    evalue=np.array(e.shap_values(XX_m[0:100]))
    shapley_cnn[1,j,:,:]=np.mean(evalue.reshape(100,28,28,1),axis=0)[:,:,0]

    for k in range(Kfold):
        teacher = keras.Sequential(
        [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(128, (3, 3), activation = 'relu', padding="same"),
        #layers.LeakyReLU(negative_slope=0.2),
        layers.MaxPooling2D(pool_size=(4, 4)),
        layers.Flatten(),
        layers.Dense(128,activation="relu"),
        layers.Dense(1,activation="sigmoid"),
        ],
        name="teacher",
        )
        teacher.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
        )
        # Train and evaluate teacher on data.
        teacher.fit(np.delete(XX_m,Obs_p_m[:,k],0), np.delete(YY_m,Obs_p_m[:,k],0), epochs=epochm[0], batch_size=batch_sizem[0],verbose=0)

        student = keras.Sequential(
        [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(5, (3, 3), activation = 'relu', padding="same"),
        #layers.LeakyReLU(negative_slope=0.2),
        layers.MaxPooling2D(pool_size=(4, 4)),
        layers.Flatten(),
        layers.Dense(1,activation="sigmoid"),
        ],
        name="student",
        )

        # comment simultaneously: ctrl+/
        # student.compile(
        #     optimizer=keras.optimizers.Adam(),
        #     loss=keras.losses.BinaryCrossentropy(),
        #     metrics=[keras.metrics.BinaryAccuracy()],
        # )
        # # Train and evaluate teacher on data.
        # student.fit(XX_m, YY_m, epochs=100 ,batch_size=32)
        # loss1, accuracy1=student.evaluate(X_test, Y_test)

        # Initialize and compile distiller
        distiller = Distiller(student=student, teacher=teacher)
        #distiller.build(input_shape=(None, 28, 28, 1))
        distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.BinaryAccuracy()],
        student_loss_fn=keras.losses.BinaryCrossentropy(),
        distillation_loss_fn=keras.losses.MeanSquaredError(),
        alpha=0.8,
        temperature=1,
        )

        # Distill teacher to student
        distiller.fit(np.delete(XX_m,Obs_p_m[:,k],0), np.delete(YY_m,Obs_p_m[:,k],0), epochs=epochm[1], batch_size=batch_sizem[1],verbose=0)
        mu_cv[Obs_p_m[:,k],0]=distiller.predict(XX_m[Obs_p_m[:,k]])[:,0]
    
    mu_mat[1,j,:,j]=mu_cv[:,0]

# Combine the arrays into a dictionary (or a list)
data_to_save = {
    "shap_cnn": shapley_cnn,
    "mu_mat": mu_mat,
    "mu_test": mu_test
}
# Save as a .pkl file
with open("cnn_arrays.pkl", "wb") as file:
    pickle.dump(data_to_save, file)


        

