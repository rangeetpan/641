x1=[0.328335
1.94333
2.595004
0.601665
1.228336
1.956663
1.33
1.94166
1.17
0.16334];
x2=[0.830001
3.150005
2.41167
1.704995
1.050001
3.55667
0.556666
1.535005
1.096664
1.655];
x4=[2.026667
3.33167
1.91833
0.976664
0.30667
2.785005
0.55167
1.478333
1.61666
1.451664];
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
legend('FMNIST-1','FMNIST-2','FMNIST-4','location','best')
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
print(fig, 'FMNIST_Delta.pdf', '-dpdf')