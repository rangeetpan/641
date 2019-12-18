x1=[0.0137667
0.00986664
0.03795
0.02680004
0.00486663
0.01479995
0.03291666
0.00555005
0.00975001
0.0084167];
x2=[-0.25833
4.025
2.608333
0.304996
0.06333
2.11
2.248336
0.288335
0.25833
1.096664];
x4=[1.228334
0.615001
3.170002
0.870002
0.93
1.021664
1.70833
1.616666
1.88167
1.115];
y=0.002:0.0005:0.0065;
x1max=max(x1);
y1max=find(x1==x1max);
y1max=0.002+(y1max-1)*0.0005;
x2max=max(x2);
y2max=find(x2==x2max);
y2max=0.002+(y2max-1)*0.0005;
x4max=max(x4);
y4max=find(x4==x4max);
y4max=0.002+(y4max-1)*0.0005;
figure
plot(y,x1,'-o','LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])
hold on
plot(y,x2,'-ks','LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',[0.5,0.5,0.5])
plot(y,x4,'-P','LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor',[0.5,0.5,0.5])
plot(y1max,x1max,'-^','LineWidth',2,...
    'MarkerSize',20,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor',[0.5,0.5,0.5])
plot(y2max,x2max,'-^','LineWidth',2,...
    'MarkerSize',20,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor',[0.5,0.5,0.5])
plot(y4max,x4max,'-^','LineWidth',2,...
    'MarkerSize',20,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor',[0.5,0.5,0.5])
legend('MNIST-1','MNIST-2','MNIST-4','location','best')
xlabel('Delta','fontweight','bold','FontSize',16)
set(gca,'XTickLabelMode','auto')
ylabel('Gain(%)','fontweight','bold','FontSize',16)
set(gca,'XLim',[0.0025 0.0065])
%set(leg,'location','best')
ax = gca;
ax.FontSize = 16; 
fig = gcf;
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig, 'MNIST_Delta.pdf', '-dpdf')