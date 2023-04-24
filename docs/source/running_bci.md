# Running a full BCI simulation

If you're eager to get started with NDS, you can run a full BCI simulation out of the box by following these steps:

1. Complete the [installation](installation.md) (make sure to install the `[extras]` dependencies).

1. Run the following script:
   ```
   run_closed_loop
   ```
   ```{note}
   You might need to give permissions like network access when running this script.
   ```

After running the `run_closed_loop` script[^1], you'll be prompted with instructions on how to perform the center-out reaching task. And that's it, you're now the `brain` that makes the `BCI`!

[^1]: This script automates running the [`center_out_reach` task](tasks.md#provided-task) that is used to generate behavior data, the [encoder](encoder.md) to consume and generate spiking rates, the [ephys_generator](ephys_generator.md) to transform the spiking rates into electrophysiology data, and the [decoder](decoders.md) to transform electrophysiology data back into decoded behavior.
