using System;
using WfdbCsharpWrapper;

namespace classifier
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("pgain Using Record Class");
            using (var record = new Record("D:\\Desarrollo\\databases\\MIT-BIH\\100"))
            {
                record.Open();
                int counter = 0;
                foreach (var signal in record)// or (var signal in record.Signals)
                {
                    Console.WriteLine("Signal {0} gain = {1}", counter, signal.Gain);
                    counter++;
                }
            }

            readRecord();
        }
        public static void readRecord()
        {
           
        }
    }
}
