using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lab3_ISIT 
{ 
    //Передаточные функции и их производные
    #region Transfer functions and their derivatives
    public enum TransferFunction
    {
        Sigmoid
    }

    class TransferFunctions
    {
        //функция активации
        public static double Evaluate(TransferFunction tFunc, double input)
        {
            switch (tFunc)
            {
                case TransferFunction.Sigmoid:
                    return sigmoid(input);
                default:
                    return 0.0;
            }
        }

        //производная функция активации
        public static double EvaluateDerivative(TransferFunction tFunc, double input)
        {
            switch (tFunc)
            {
                case TransferFunction.Sigmoid:
                    return sigmoid_derivative(input);
                default:
                    return 0.0;
            }
        }

        //Функция активации — способ нормализации входных данных (вывод в необходимый диапазон значений)
        private static double sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        //производная от функции активации
        private static double sigmoid_derivative(double x)
        {
            return sigmoid(x) * (1 - sigmoid(x));
        }
#endregion
    }

    /// <summary>
    /// метод обратного распространения
    /// </summary>
    public class BackPropagationNetwork 
    {
        #region Private data

        private int layerCount; //счетчик слоев
        private int inputSize; //размерность входа
        private int[] layerSize;//размерность слоя
        private TransferFunction[] transferFunction; //функция активации


        private double[][] layerOutput; //выход слоя (для переборов в циклах)
        private double[][] layerInput; //вход слоя (для переборов в циклах)
        private double[][] bias; /*Нейрон смещения нужен для того, чтобы иметь возможность получать 
                                 выходной результат, путем сдвига графика функции активации вправо или влево. 
                                 их можно размещать на входном слое и всех скрытых слоях, но никак не на выходном слое, 
                                 так как им попросту не с чем будет формировать связь. 
                                 Синапсов между двумя bias нейронами быть не может. https://habr.com/post/313216/ */

        private double[][] delta; //разница
        private double[][] previousBiasDelta;//переменная для настройки нейронов смещения


        private double[][][] weight; //вес синапса
        private double[][][] previousWeightDelta;//переменная для настройки весов

        #endregion
        #region Constructors
        public BackPropagationNetwork(int[] layerSizes, TransferFunction[] transferFunctions)
        {
            //Инициализация слоев нейронов
            layerCount = layerSizes.Length - 1; //счетчик слоев 
            inputSize = layerSizes[0];
            layerSize = new int[layerCount];

            for (int i = 0; i < layerCount; i++)
            {
                layerSize[i] = layerSizes[i + 1];
            }

            transferFunction = new TransferFunction[layerCount];

            //Определение размеров массивов
            bias = new double[layerCount][];
            previousBiasDelta = new double[layerCount][];
            delta = new double[layerCount][];
            layerOutput = new double[layerCount][];
            layerInput = new double[layerCount][];

            weight = new double[layerCount][][];
            previousWeightDelta = new double[layerCount][][];

            //Заполнение двумерных массивов
            for (int l = 0; l < layerCount; l++)
            {
                bias[l] = new double[layerSize[l]];
                previousBiasDelta[l] = new double[layerSize[l]];
                delta[l] = new double[layerSize[l]];
                layerOutput[l] = new double[layerSize[l]];
                layerInput[l] = new double[layerSize[l]];

                weight[l] = new double[l == 0 ? inputSize : layerSize[l - 1]][];
                previousWeightDelta[l] = new double[l == 0 ? inputSize : layerSize[l - 1]][];

                for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
                {
                    weight[l][i] = new double[layerSize[l]];
                    previousWeightDelta[l][i] = new double[layerSize[l]];
                }
            }

            //Инициализация нейронов смещения (байесов) на слоях
            for (int l = 0; l < layerCount; l++)
            {
                for (int j = 0; j < layerSize[l]; j++)
                {
                    bias[l][j] = Gaussian.GetRandomGaussian(); //случайным методом
                    previousBiasDelta[l][j] = 0.0;
                    layerOutput[l][j] = 0.0;
                    layerInput[l][j] = 0.0;
                    delta[l][j] = 0.0;
                }

                //Инициализация весов на слоях
                for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
                {
                    for (int j = 0; j < layerSize[l]; j++)
                    {
                        weight[l][i][j] = Gaussian.GetRandomGaussian(); //случайным методом
                        previousWeightDelta[l][i][j] = 0.0;
                    }
                }
            }
        }

        #endregion
        #region Methods

        public void Run(ref double[] input, out double[] output)
        {
            output = new double[layerSize[layerCount - 1]];

            /*Запуск сети*/
            for (int l = 0; l < layerCount; l++)
            {
                for (int j = 0; j < layerSize[l]; j++)
                {
                    double sum = 0.0;
                    for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
                    {
                        sum += weight[l][i][j] * (l == 0 ? input[i] : layerOutput[l - 1][i]);
                    }

                    sum += bias[l][j];
                    layerInput[l][j] = sum;

                    layerOutput[l][j] = TransferFunctions.Evaluate(transferFunction[l], sum);
                }
            }

            // Копирование выхода в выходной массив
            for (int i = 0; i < layerSize[layerCount - 1]; i++)
            {
                output[i] = layerOutput[layerCount - 1][i];
            }
        }

        public double Train(ref double[] input, ref double[] desired, double TrainingRate/*скорость обучения*/, double Impulse/*момент*/)
        {
            double error = 0.0, sum = 0.0, weightDelta = 0.0, biasDelta = 0.0;
            double[] output = new double[layerSize[layerCount - 1]];

            //Запуск сети
            Run(ref input, out output);

            //Обратное распространение ошибки
            for (int l = layerCount - 1; l >= 0; l--) //начинаем с выходного слоя
            {
                //для выходноого слоя (отсутсвуют исходящие синапсы)
                if (l == layerCount - 1)
                {
                    //считаем разницу между желаемым и полученным результатом и 
                    //умножаем на производную функции активации от входного значения данного нейрона на данном слое
                    //https://habr.com/post/313216/
                    
                    for (int k = 0; k < layerSize[l]; k++)
                    {
                        delta[l][k] = output[k] - desired[k]; //разница действительное-желаемое значение 
                        error += 0.5* Math.Pow(delta[l][k], 2); //ошибка сети на данном этапе
                        delta[l][k] *= TransferFunctions.EvaluateDerivative(transferFunction[l], layerInput[l][k]); //домножили на производную, дельта посчитана
                    }
                }

                //для скрытого слоя
                /*суть МОР заключается в том чтобы распространить ошибку 
                 * выходных нейронов на все веса НС. Ошибку можно вычислить только на выходном уровне, 
                 * как мы это уже сделали, также мы вычислили дельту в которой уже есть эта ошибка. 
                 * Следственно теперь мы будем вместо ошибки использовать дельту, 
                 * которая будет передаваться от нейрона к нейрону*/
                else
                {
                    for (int i = 0; i < layerSize[l]; i++)
                    {
                        sum = 0.0;
                        for (int j = 0; j < layerSize[l + 1]; j++)
                        {
                            sum += weight[l + 1][i][j] * delta[l][j];
                        }
                        sum *= TransferFunctions.EvaluateDerivative(transferFunction[l], layerInput[l][i]);

                        delta[l][i] = sum;
                    }
                }
            }

            // Обновление весов
            for (int l = 0; l < layerCount; l++) //перебор по слоям
            {
                for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++) 
                {
                    //вес = скорость обучения * дельту

                    for (int j = 0; j < layerSize[l]; j++)
                    {
                        weightDelta = TrainingRate * delta[l][j] * (l == 0 ? input[i] : layerOutput[l - 1][i]); //delta[l][j] * (l == 0 ? input[i] : layerOutput[l - 1][i] --- ГРАДИЕНТ,
                                                                                                                //первый множитель - точка в начале синапса, второй - в конце
                        weight[l][i][j] -= weightDelta + Impulse * previousWeightDelta[l][i][j]; //обновление значения веса, где Impulse - момент; 
                                                                                                 //величина после знака равно и есть искомая величина, на которую необходимо изменить вес

                        previousWeightDelta[l][i][j] = weightDelta; //текущая дельта становится предыдущей для послед расчета
                    }
                }
            }

            //байесов(Н-ы смещения)
            //Иногда на схемах не обозначают нейроны смещения, а просто учитывают их веса при вычислении входного значения
            //аналогиный рассчет; нейроны смещения могут, либо присутствовать в нейронной сети по одному на слое, либо полностью отсутствовать
            //синапсов между двумя bias нейронами быть не может!
            for (int l = 0; l < layerCount; l++)
            {
                for (int i = 0; i < layerSize[l]; i++)
                {
                    biasDelta = TrainingRate * delta[l][i]; 
                    bias[l][i] -= biasDelta + Impulse * previousBiasDelta[l][i]; //обновление значения байеса, где Impulse - момент; 

                    previousBiasDelta[l][i] = biasDelta; //текущая байес-дельта становится предыдущей для послед расчета
                }
            }

            //возвращаем ошибку
            return error;
        }
        #endregion
    }

    public static class Gaussian //для инициализации случайных значений, распредление гаусса, ИСТОЧНИК: https://repl.it/@HHeld/Neural-Network
    {
        private static Random gen = new Random();

        public static double GetRandomGaussian()
        {
            return GetRandomGaussian(0.0, 1.0);
        }

        public static double GetRandomGaussian(double mean, double stddev)
        {
            double rVal1, rVal2;

            GetRandomGaussian(mean, stddev, out rVal1, out rVal2);

            return rVal1;
        }

        public static void GetRandomGaussian(double mean, double stddev, out double val1, out double val2)
        {
            double u, v, s, t;
            do
            {
                u = 2 * gen.NextDouble() - 1;
                v = 2 * gen.NextDouble() - 1;

            } while (u * u + v * v > 1 || (u == 0 && v == 0));

            s = u * u + v * v;
            t = Math.Sqrt((-2.0 * Math.Log(s)) / s);

            val1 = stddev * u * t + mean;
            val2 = stddev * v * t + mean;
        }
    }
}
