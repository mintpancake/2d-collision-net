import sampler
import data_generator
import trainer
import plot
import test_data_generator
import tester
import visualizer


def run():
    sampler.run()
    data_generator.run()
    trainer.run()
    plot.run()
    test_data_generator.run()
    tester.run()
    visualizer.run()


def run_without_regeneration():
    trainer.run()
    plot.run()
    test_data_generator.run()
    tester.run()
    visualizer.run()


def run_whatever():
    plot.run()
    test_data_generator.run()
    tester.run()
    visualizer.run()

if __name__ == "__main__":
    run_whatever()
