using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


//Нейросети: https://habr.com/post/312450/, https://habr.com/post/313216/, https://repl.it/@HHeld/Neural-Network

//ЗАДАНИЕ
//9. Реализуйте с помощью нейронной сети операцию умножения трех чисел из диапазона [0, 1]. hh

namespace Lab3_ISIT
{
    class Program
    {
        static void Main(string[] args)
        {
            //задаем параметры сети --- 3 слоя: 3 вх нейрона, 6 скрытых нейронов, 1 вых нейрон
            int[] layerSizes = new int[3] { 3, 6, 1 }; 

            //активационная функция лог-сигмоидная
            TransferFunction[] tFuncs = new TransferFunction[1]
            {
                TransferFunction.Sigmoid,
            };

            //задействуем метод обратного распространения
            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, tFuncs); //слои, функц.активации
            
            //объявялем перменные для 3х чисел и их произведения
            double a, b, c, multy; 
          
            Console.WriteLine("Введите первое число из диапазона [0;1]\n");
            a = double.Parse(Console.ReadLine());

            Console.WriteLine("\nВведите второе число из диапазона [0;1]]\n");
            b = double.Parse(Console.ReadLine());
            Console.WriteLine("\nВведите третье число из диапазона [0;1]\n");
            c = double.Parse(Console.ReadLine()); 

            multy = a * b * c; //перемножение
            Console.WriteLine("\nПроизведение чисел a, b, c = " + multy + "\n");

            double[] input = new double[3] { a, b, c }; //вход 3 числа
            double[] desired = new double[1] { multy }; //желаемое значение
            double[] output = new double[1]; //выход 1 число

            double error = 0.0; //инициализируем ошибку

            for (int i = 0; i < 10100; i++) // итерации
            {
                error = bpn.Train(ref input, ref desired, 0.15, 0.1);//задаем скорость распространения и момент (для градиентного спуска)
                bpn.Run(ref input, out output);
                if (i % 100 == 0) //интервал для отображения  итерации
                    Console.WriteLine("Итерация {0}:\tВход 1 {1:0.0000000} Вход 2 {2:0.0000000} "+
                        "Вход 3 {3:0.0000000} Выход {4:0.0000000} Ошибка {5:0.0000000}", i, input[0], input[1], input[2], output[0], error);
            }
            Console.ReadLine();
        }
    }
}
