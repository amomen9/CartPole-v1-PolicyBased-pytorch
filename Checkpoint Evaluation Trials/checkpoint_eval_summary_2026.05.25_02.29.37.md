\begin{table}[t]
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\caption{Statistical factors (truncation=20,000)}
\label{tab:results_summary_20000}
\begin{tabular}{l r r r r r }
\toprule
\textbf{Algorithm} & \textbf{Mean}& \textbf{Std}& \textbf{Q1}& \textbf{Median}& \textbf{Q3}\\
\midrule
A2C (softmax) & 14,508.48 & 6,830.46 &  7,918.00 & 20,000.00 & 20,000.00 \\
A2C (argmax)  &  9,738.85 & 8,164.93 &  1,141.00 &  6,328.50 & 20,000.00 \\
DQN (softmax) & 15,336.31 & 6,419.59 & 10,602.00 & 20,000.00 & 20,000.00 \\
DQN (argmax)  & 20,000.00 &     0.00 & 20,000.00 & 20,000.00 & 20,000.00 \\
PPO (softmax) & 16,998.34 & 5,581.71 & 16,528.50 & 20,000.00 & 20,000.00 \\
PPO (argmax)  & 20,000.00 &     0.00 & 20,000.00 & 20,000.00 & 20,000.00 \\
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\end{table}
