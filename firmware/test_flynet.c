/* test_flynet.c — smoke-тест выгруженного мозга: собираем и проверяем, что он живой.
 *
 *   cc -O2 -o /tmp/flynet_test firmware/test_flynet.c -lm && /tmp/flynet_test
 *   cc -O2 -DFLYNET_HEADER='"../firmware/flynet_1024.h"' ... (для контура побольше)
 *
 * Печатает три сценария сенсоров. Числа должны совпадать с тем, что даёт
 * robot/flybrain.py на тех же входах — это и есть проверка, что порт честный.
 */
#ifndef FLYNET_HEADER
#define FLYNET_HEADER "flynet_256.h"
#endif
#include FLYNET_HEADER

#include <stdio.h>

static void scenario(const char *name, flynet_t *b,
                     float odor_l, float odor_r, float loom_l, float loom_r)
{
    flynet_init(b);
    for (int tick = 0; tick < 40; tick++) {           /* 40 × 50 мс = 2 с */
        flynet_set_input(b, odor_l, odor_r, loom_l, loom_r);
        for (int i = 0; i < FLYNET_STEPS; i++)
            flynet_step(b);
    }
    float left, right;
    flynet_wheels(b, &left, &right);
    int spikes = 0;
    double rate_sum = 0.0;
    for (int i = 0; i < FLYNET_N; i++) {
        spikes += b->spike[i] ? 1 : 0;
        rate_sum += b->rate[i];
    }
    printf("%-22s odor %.1f/%.1f loom %.1f/%.1f -> left %+.6f right %+.6f "
           "| спайков %4d, сумма частот %.6f\n",
           name, odor_l, odor_r, loom_l, loom_r, left, right, spikes, rate_sum);
#ifdef FLYNET_DUMP
    {
        int worst = 0;
        double worst_x = 0.0;
        double s0 = fly_wout[0][FLYNET_FEAT], s1 = fly_wout[1][FLYNET_FEAT];
        for (int j = 0; j < FLYNET_FEAT; j++) {
            double x = ((double)b->rate[fly_feat[j]] - fly_mu[j]) / fly_sd[j];
            s0 += (double)fly_wout[0][j] * x;
            s1 += (double)fly_wout[1][j] * x;
            if (fabs(x) > fabs(worst_x)) { worst_x = x; worst = j; }
        }
        printf("    max |x| = %.6f при j=%d (feat=%d, rate=%.9f, mu=%.9f, sd=%.9f)\n",
               worst_x, worst, fly_feat[worst], b->rate[fly_feat[worst]],
               fly_mu[worst], fly_sd[worst]);
        printf("    без клипа: left %.6f right %.6f\n", s0, s1);
        printf("    wout[0][0..2]=%.6f %.6f %.6f  bias0=%.6f bias1=%.6f\n",
               fly_wout[0][0], fly_wout[0][1], fly_wout[0][2],
               fly_wout[0][FLYNET_FEAT], fly_wout[1][FLYNET_FEAT]);
    }
#endif
}

int main(void)
{
    flynet_t brain;
    printf("контур: %d нейронов, %d синапсов, читатель %d входов\n",
           FLYNET_N, FLYNET_SYN, FLYNET_FEAT);
    printf("flash %u Б, RAM %u Б\n",
           (unsigned)FLYNET_FLASH_BYTES, (unsigned)FLYNET_RAM_BYTES);

    scenario("запах слева", &brain, 0.8f, 0.2f, 0.0f, 0.0f);
    scenario("запах справа", &brain, 0.2f, 0.8f, 0.0f, 0.0f);
    scenario("препятствие справа", &brain, 0.5f, 0.5f, 0.0f, 0.7f);
    scenario("препятствие слева", &brain, 0.5f, 0.5f, 0.7f, 0.0f);
    return 0;
}
